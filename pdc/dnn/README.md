# Distributed Deep Learning (PDC)

Examples and slides on scaling DNN training across GPUs: TensorFlow data parallelism, PyTorch pipeline parallelism with mixed precision, and a guest talk on Horovod.

## `tensorflow_1_node-n_gpu/`: data parallelism on one node

| File | Description |
|------|-------------|
| `mnist_parallel.py` | Trains a deliberately wide MLP (Dense 10000/9000/10000/256/256) on MNIST with `tf.distribute.MirroredStrategy`. The global batch is 64 × the number of replicas. The model is built and compiled inside `strategy.scope()`. Weights are saved with `ModelCheckpoint` to `training_1/`. The same script is in `hpml/data_parallel/1_node/keras/mnist_multigpu.py`. |
| `mnist_parallel.job` | A SLURM batch script for the `dgxa100` partition. Set `--gres` for the number of GPUs you want and fill in `--mail-user`. |
| `output-dgx.txt` | A sample run on an 8 × A100 DGX: device listing, then 2 epochs reaching about 97% test accuracy. |
| `data_parallel-handson.pptx` / `.pdf` | Hands-on slides: `tf.distribute` strategies (Mirrored, MultiWorkerMirrored, CentralStorage, ParameterServer), cross-device ops (`NcclAllReduce`, `HierarchicalCopyAllReduce`, `ReductionToOneDevice`), and cluster commands. |
| `cat.jpg` | An image asset for the hands-on. |

```bash
python mnist_parallel.py       # or: sbatch mnist_parallel.job
```

## `pytorch/`: pipeline parallelism and mixed precision

`pipeline_plus_mixedprecision.py` trains an AlexNet with a **mixture-of-experts (MoE)** classifier head on CIFAR-100, split across **2 GPUs** with FairScale `Pipe` (GPipe-style micro-batching, `chunks=4`). It uses automatic mixed precision.

| Function / class | Role |
|------------------|------|
| `get_train_valid_loader`, `get_test_loader` | CIFAR-100 loaders: resize to 227, optional crop and flip augmentation, 90/10 train/validation split |
| `AlexNetMoELossFree` | Conv trunk → FC layers → a gating layer that picks **one** expert per sample (top-1). `expert_bias` is a buffer added to the gate scores. |
| `update_expert_bias()` | **Loss-free load balancing.** After each step, the bias is lowered by γ for overloaded experts and raised for underused ones. No auxiliary loss is needed. |
| `MoEStage1`, `MoEStage2` | The two pipeline stages: trunk + FC on `cuda:0`, and gating + experts on `cuda:1` (unbalanced on purpose, as a teaching example). |
| `build_moe_pipeline()` | Wraps the stages in `Pipe(balance=[1,1], devices=[cuda:0, cuda:1], chunks=4)`. |
| `evaluate()` | Accuracy, with inputs on `cuda:0` and labels on `cuda:1`. |
| `main()` | SGD and `CrossEntropyLoss`, with `autocast()` for the forward pass and `GradScaler` for scaled backward and optimizer steps. |

Requirements: `torch`, `torchvision`, `fairscale`, and 2 GPUs. The defaults (batch size 4096, 100 epochs) are heavy, so reduce them for a quick demo. Newer PyTorch versions warn that `torch.cuda.amp` is deprecated in favour of `torch.amp`.

## `horovod/`: guest slides

`horovod_iqrah.pptx` is a talk by Jurdana Masuma Iqrah (UTSA) on distributed U-Net training for polar sea-ice classification on Sentinel-2 imagery. It covers Horovod's ring all-reduce, `hvd.init()`, `hvd.DistributedOptimizer`, and scaling results on a DGX A100. The folder has slides only, no code.
