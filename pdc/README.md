# Parallel and Distributed Computing (PDC)

Code examples for the PDC course: shared-memory programming (Pthreads, OpenMP), distributed-memory programming (MPI), GPU programming (CUDA, OpenMP offload), debugging, and distributed deep learning. Every source folder has its own README describing its examples.

| Topic | Folder | Contents |
|-------|--------|----------|
| Debugging with gdb | [`gdb_src/`](gdb_src) | Four C programs with planted bugs for gdb practice |
| POSIX threads | [`pthreads/src/`](pthreads/src) | Create, join, cancel and detach threads; mutexes, barriers, condition variables, thread-local data; Monte Carlo π and false sharing |
| OpenMP, part 1 | [`openmp/src1/`](openmp/src1) | Parallel regions, race conditions, `critical`/`atomic`, `omp for` |
| OpenMP, part 2 | [`openmp/src2/`](openmp/src2) | One program per directive or clause: `sections`, `single`, `task`, `private`/`firstprivate`/`lastprivate`, `reduction`, `schedule`, `nowait` |
| OpenMP GPU offload | [`openmp/src1.5/`](openmp/src1.5) | `target teams` versions of the CUDA matrix-multiply and reduction kernels |
| MPI | [`mpi/src/`](mpi/src) | Point-to-point, collectives, non-blocking communication, groups and communicators, custom datatypes and operations |
| CUDA basics | [`cuda/src/`](cuda/src) | Kernels, thread/block indexing, vector add, shared-memory dot product, CPU vs GPU matrix multiply |
| CUDA matrix multiply | [`cuda/src1.5/`](cuda/src1.5) | From one block to multiple blocks to shared-memory tiling to register tiling; reduction; coalescing benchmark |
| CUDA advanced | [`cuda/src2/`](cuda/src2) | NVIDIA samples: streams, multi-copy overlap, multi-GPU, zero-copy, occupancy, cooperative groups |
| CUDA debugging and profiling | [`cuda/cuda_gdb/`](cuda/cuda_gdb) | `cuda-gdb` demo programs and Nsight Systems reports |
| Distributed DNN training | [`dnn/`](dnn) | TensorFlow MirroredStrategy, PyTorch pipeline parallelism with mixed precision, Horovod slides |

## Toolchain

| Folder type | Compiler | Run |
|-------------|----------|-----|
| Pthreads / OpenMP / gdb | `gcc` (`-pthread` / `-fopenmp`) | `./prog`, with `OMP_NUM_THREADS` for OpenMP |
| MPI | `mpicc` (Open MPI or MPICH) | `mpirun -np 4 ./prog` |
| CUDA | `nvcc` (CUDA Toolkit) | `./prog` on a machine with an NVIDIA GPU |
| OpenMP offload | GCC with NVPTX offload (`-foffload=nvptx-none`) | `OMP_TARGET_OFFLOAD=MANDATORY ./prog` |

Most C/CUDA folders have a `Makefile`, so `make` builds every example in the folder and `make clean` removes the binaries.
