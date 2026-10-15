# High-Performance Machine Learning (HPML)

> **Animated walkthrough:** step through every example here, with the matching source lines highlighted, in the [Visual Lab: HPML course](https://buddhi1.github.io/pdc-class/#/hpml).

Code for the HPML course. It covers GPU programming and performance for ML workloads, an introduction to TensorFlow, and scaling training across GPUs and nodes. Every source folder has its own README describing its examples.

| Topic | Folder | Contents |
|-------|--------|----------|
| CUDA fundamentals | [`cuda/src1/`](cuda/src1) | Kernels and indexing, shared-memory vs atomic dot product, serial/OpenMP/GPU/tiled matrix multiply, multi-GPU matrix multiply (OpenMP threads vs streams) |
| CUDA performance | [`cuda/src2/`](cuda/src2) | Warp divergence, memory coalescing and shared-memory bank-conflict benchmarks, with Nsight Compute metrics |
| TensorFlow intro | [`tensorflow/`](tensorflow) | Colab notebooks: tensors, a first neural network, CNNs, transfer learning |
| Data-parallel training | [`data_parallel/`](data_parallel) | MirroredStrategy, CentralStorage, Keras 3 on JAX (data and model parallel), PyTorch DDP, MultiWorkerMirrored (custom loop), ParameterServer, input sharding |
| Homework | [`../hpml26-dataparallel_hw/`](../hpml26-dataparallel_hw) | VGG16 / CIFAR-100 MirroredStrategy job for NCSA Delta |

## Material shared with PDC

- [`pdc/dnn/tensorflow_1_node-n_gpu/`](../pdc/dnn/tensorflow_1_node-n_gpu): the "Data Parallel DNN Training with TensorFlow" hands-on slides and `mnist_parallel.py` (the same script as `data_parallel/1_node/keras/mnist_multigpu.py`).
- [`pdc/dnn/pytorch/`](../pdc/dnn/pytorch): pipeline parallelism with mixed precision.
- [`pdc/cuda/src1.5/`](../pdc/cuda/src1.5): step-by-step CUDA matrix-multiply kernels, useful alongside `cuda/src1`.
- `cuda/src1` is an extended version of [`pdc/cuda/src`](../pdc/cuda/src). It adds better timing, `-O3` builds and the multi-GPU examples.

## Environment

- **CUDA:** the `nvcc` CUDA Toolkit. `ncu` (Nsight Compute) for the profiling exercises. Two or more GPUs for the multi-GPU programs.
- **Python:** see [`data_parallel/commandsForArc.txt`](data_parallel/commandsForArc.txt) for the conda setup. Typical packages: `tensorflow[and-cuda]`, `tf_keras`, `keras`, `jax[cuda]`, `torch`, `tensorflow_datasets`.
