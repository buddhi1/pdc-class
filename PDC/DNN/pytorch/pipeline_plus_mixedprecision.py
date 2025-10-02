#!/usr/bin/env python
import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from fairscale.nn import Pipe
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

# CIFAR-100 dataset loader with data augmentation and normalization
# This code is adapted from the torchvision CIFAR-100 example
# and modified to include a mixture of experts (MoE) model
def get_train_valid_loader(data_dir, batch_size, augment, random_seed, valid_size=0.1, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409],
        std=[0.2673, 0.2564, 0.2761],
    )
    valid_transform = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(227),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )
    valid_dataset = datasets.CIFAR100(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=valid_sampler)
    return train_loader, valid_loader


def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409],
        std=[0.2673, 0.2564, 0.2761],
    )
    transform_pipeline = transforms.Compose([
        transforms.Resize((227, 227)),
        transforms.ToTensor(),
        normalize,
    ])
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False,
        download=True, transform=transform_pipeline,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return test_loader

@torch.no_grad()
def evaluate(model, loader, device0="cuda:0", device1="cuda:1"):
    """Run model on loader and return accuracy (%)"""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device0)
        labels = labels.to(device1)
        outputs = model(images)
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    model.train()
    return 100.0 * correct / total

# Define the AlexNet model with a mixture of experts (MoE) architecture
class AlexNetMoELossFree(nn.Module):
    def __init__(self, num_classes=100, num_experts=3, expert_hidden=4096):
        super().__init__()
        # --- conv trunk ---
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(96, 256, 5, padding=2),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, 3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.BatchNorm2d(384), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )
        self.flatten_dim = 256 * 6 * 6

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.flatten_dim, expert_hidden),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(expert_hidden, expert_hidden),
            nn.ReLU(),
        )

        self.num_experts = num_experts
        self.gating      = nn.Linear(expert_hidden, num_experts)
        self.experts     = nn.ModuleList([
            nn.Linear(expert_hidden, num_classes)
            for _ in range(num_experts)
        ])

        self.register_buffer("expert_bias", torch.zeros(num_experts))
        self.gamma = 0.001

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        h = self.fc(x)

        gate_scores = self.gating(h)                    # (B, E)
        biased      = gate_scores + self.expert_bias   # (B, E)
        chosen      = torch.argmax(biased, dim=1)       # (B,)

        out = torch.stack([
            self.experts[chosen[i]](h[i])
            for i in range(h.size(0))
        ], dim=0)                                       # (B, num_classes)

        self._last_chosen = chosen
        return out

    @torch.no_grad()
    def update_expert_bias(self):
        chosen = self._last_chosen
        B = chosen.size(0)
        expected = B / float(self.num_experts)
        counts = torch.zeros_like(self.expert_bias)
        for idx in chosen:
            counts[idx] += 1
        for i in range(self.num_experts):
            if counts[i] > expected:
                self.expert_bias[i] -= self.gamma
            elif counts[i] < expected:
                self.expert_bias[i] += self.gamma

#######################################################
# 2) Split into two pipeline stages for FairScale Pipe(Gpipe, https://arxiv.org/abs/1811.06965)
# Define stages 1 to n here where n is the number of  
# devices used in the pipeline. Each stage will   
# contain a subset of the model's layers. in mine, 
# stage 1 contains the feature extractor and the first 
# fully connected layer, and stage 2 contains the gating 
# and experts. The output of stage 1 is the input to stage 2. 
# this is kind of dumb because that first stage is much heavier
# than the second stage, but this is an example of how to do it.
#######################################################
class MoEStage1(nn.Module):
    def __init__(self, moe_model: AlexNetMoELossFree):
        super().__init__()
        self.features = moe_model.features
        self.fc       = moe_model.fc

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        h = self.fc(x)
        return h

class MoEStage2(nn.Module):
    def __init__(self, moe_model: AlexNetMoELossFree):
        super().__init__()
        self.gating      = moe_model.gating
        self.experts     = moe_model.experts
        self.register_buffer("expert_bias", moe_model.expert_bias.clone())
        self.gamma       = moe_model.gamma

    def forward(self, h):
        gate_scores = self.gating(h)
        biased      = gate_scores + self.expert_bias
        chosen      = torch.argmax(biased, dim=1)

        out = torch.stack([
            self.experts[chosen[i]](h[i])
            for i in range(h.size(0))
        ], dim=0)

        self._last_chosen = chosen
        return out

    @torch.no_grad()
    def update_expert_bias(self):
        chosen = self._last_chosen
        B = chosen.size(0)
        expected = B / float(len(self.expert_bias))
        counts = torch.zeros_like(self.expert_bias)
        for idx in chosen:
            counts[idx] += 1
        for i in range(len(self.expert_bias)):
            if counts[i] > expected:
                self.expert_bias[i] -= self.gamma
            elif counts[i] < expected:
                self.expert_bias[i] += self.gamma

def build_moe_pipeline(num_gpus=2, chunks=4, **moe_kwargs):
    base = AlexNetMoELossFree(**moe_kwargs)
    s1   = MoEStage1(base).to("cuda:0")
    s2   = MoEStage2(base).to("cuda:1")

    # Ensure the stages are on the correct devices
    model = nn.Sequential(s1, s2)
    # Ensure the model is on the correct device
    pipe  = Pipe(
        model,
        balance=[1, 1],
        devices=[f"cuda:{i}" for i in range(num_gpus)],
        chunks=chunks,
    )
    return base, pipe, s2

def main():
    data_dir   = "./data"
    batch_size = 4096
    num_epochs = 100
    lr         = 5e-3

    train_loader, valid_loader = get_train_valid_loader(
        data_dir, batch_size, augment=True, random_seed=42
    )
    test_loader = get_test_loader(data_dir, batch_size)

    base_model, pipe_model, stage2 = build_moe_pipeline(
        num_classes=100, num_experts=4, expert_hidden=4096
    )

    optimizer = optim.SGD(pipe_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    scaler    = GradScaler()  

    pipe_model.train()

    start = time.time()

    scaler = GradScaler()  
    for epoch in range(num_epochs):
        
        # Reset the expert bias at the start of each epoch
        for i, (images, labels) in enumerate(train_loader):
            images = images.to("cuda:0")
            labels = labels.to("cuda:1")
            
            optimizer.zero_grad()
            with autocast():
                outputs = pipe_model(images)
                loss = criterion(outputs, labels)
            # Scale the loss and backpropagate
            # This is where mixed precision comes into play
            # The scaler will handle the scaling of gradients
            # and the optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            stage2.update_expert_bias()

            if (i + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, "
                      f"Batch {i+1}/{len(train_loader)}, "
                      f"Loss {loss.item():.4f}")

        
        val_acc = evaluate(pipe_model, valid_loader)
    train_time = time.time() - start
    print(f"done in {train_time:.1f}s — val acc: {val_acc:.2f}%\n")

    test_acc = evaluate(pipe_model, test_loader)
    print(f"Test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    main()
