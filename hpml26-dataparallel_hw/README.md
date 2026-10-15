# Data-Parallel Homework (HPML 2026)

Files for the data-parallel training assignment: a single-node, multi-GPU `MirroredStrategy` run of VGG16 on CIFAR-100.

| File | Description |
|------|-------------|
| `mirrored.py` | VGG16 built from scratch (`weights=None`) with a Flatten → Dense 512 → Dropout → Dense 100 head. CIFAR-100 images are resized to 224×224 and preprocessed for VGG. The global batch is 32 × the number of replicas (`drop_remainder=True`). Training uses SGD with momentum for 30 epochs × 400 steps, saves a checkpoint every epoch to `./mirrored_ckpt`, and finishes with an evaluation on the test set. Legacy Keras 2 is forced with `TF_USE_LEGACY_KERAS=1`. |
| `mirrored.sh` | SLURM script for NCSA Delta: `gpuA100x4` partition, 1 node with 4 GPUs, 48 GB memory, 30 minutes. Loads `tensorflow-conda` and `aws-ofi-nccl`, then runs `mirrored.py`. Change `--account` to your own allocation. |

```bash
sbatch mirrored.sh        # writes output to tf_mirror.out and tf_mirror.err
```

## Key functions in `mirrored.py`

- `preprocess(x, y)` resizes images, applies `vgg16.preprocess_input` and squeezes the labels.
- `make_dataset()` builds the training pipeline (shuffle → map → repeat → batch → prefetch) and the test pipeline. `MirroredStrategy` splits each global batch across the GPUs.

See [`hpml/data_parallel`](../hpml/data_parallel) for the background examples (MirroredStrategy, multi-worker, parameter server).
