'''
conda create -n keras_torch310 python=3.10 -y
conda activate keras_torch310
pip install torch torchvision torchaudio
'''

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

# 1. Define the identical Simple CNN Model
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28 * 1, 200)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(200, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)

# --- NEW: The Inference Function ---
def load_and_predict(filepath="ddp_survival_backup.pt"):
    """Loads a saved state_dict from disk and runs inference."""
    print(f"\n--- Loading Model Weights from {filepath} ---")
    
    # 1. Rebuild the blank architecture (Mandatory in PyTorch!)
    model = SimpleCNN()
    
    # 2. Load the checkpoint dictionary from the hard drive
    # We use map_location='cpu' so we can safely run inference on a normal machine
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    
    # 3. Pour the saved weights into the blank model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 4. CRITICAL: Set the model to evaluation mode (disables Dropout/BatchNorm)
    model.eval()
    
    # 5. Create a dummy test image (Batch, Channels, Height, Width)
    test_image = torch.randn(1, 1, 28, 28)
    
    # 6. Run inference without calculating gradients (saves memory/compute)
    print("Running prediction...")
    with torch.no_grad():
        predictions = model(test_image)
        
    predicted_class = torch.argmax(predictions, dim=1).item()
    print(f"Raw Probabilities: \n{predictions}")
    print(f"Predicted Class: {predicted_class}")


# 2. Setup the NCCL Communication Group
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    torch.distributed.destroy_process_group()

# 3. The Training Loop
def train(rank, world_size):
    setup(rank, world_size)
    print(f"[GPU {rank}] Process initialized and connected.")

    checkpoint_path = "ddp_survival_backup.pt"
    start_epoch = 0

    model = SimpleCNN().to(rank)
    optimizer = optim.Adam(model.parameters())

    # The Restore Logic
    if os.path.exists(checkpoint_path):
        print(f"[GPU {rank}] Found backup! Restoring state...")
        loc = f'cuda:{rank}'
        checkpoint = torch.load(checkpoint_path, map_location=loc, weights_only=False)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1 
        print(f"[GPU {rank}] Resuming from Epoch {start_epoch}")

    # Wrap the model in DDP AFTER loading weights
    ddp_model = DDP(model, device_ids=[rank])

    # Create dummy data
    inputs = torch.randn(128, 1, 28, 28)
    labels = torch.randn(128, 10)
    dataset = TensorDataset(inputs, labels)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
    loss_fn = nn.MSELoss()

    print(f"[GPU {rank}] Starting training...")
    
    total_epochs = 5
    for epoch in range(start_epoch, total_epochs):
        sampler.set_epoch(epoch) 
        
        for batch_inputs, batch_labels in dataloader:
            batch_inputs = batch_inputs.to(rank)
            batch_labels = batch_labels.to(rank)

            optimizer.zero_grad()
            outputs = ddp_model(batch_inputs)
            loss = loss_fn(outputs, batch_labels)
            
            loss.backward()
            optimizer.step()
            
        # The Backup Logic (Rank 0 Only)
        if rank == 0:
            print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': ddp_model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, checkpoint_path)
            print(f"--> Rank 0 saved survival backup to {checkpoint_path}")

    cleanup()

def main():
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print("This script requires at least 2 GPUs.")
        return

    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
    print("PyTorch DDP Demo Complete!")
    
    # =====================================================================
    # TEST THE SAVED MODEL
    # Uncomment the line below to test loading the model and making a prediction!
    # =====================================================================
    # load_and_predict("ddp_survival_backup.pt")

if __name__ == "__main__":
    main()