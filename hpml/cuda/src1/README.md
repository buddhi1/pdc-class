# CUDA Fundamentals and Multi-GPU Matrix Multiply (HPML)

The HPML version of the CUDA basics, extended with timing done properly and two multi-GPU matrix multiplies. Most files are close variants of [`pdc/cuda/src`](../../../pdc/cuda/src). The differences are noted below.

## Build and run

```bash
make                         # nvcc -O3; matMul and matMulMultiGPU also get -Xcompiler -fopenmp
OMP_NUM_THREADS=8 ./matMul
./matMulMultiGPU             # needs 2 or more GPUs
make clean
```

## Basics

| File | What it shows |
|------|---------------|
| `add.cu` | A one-thread kernel adds two ints. Shows the round trip through `cudaMalloc` and `cudaMemcpy`. |
| `helloworld.cu` | Device `printf` with `<<<2,10>>>`. `cudaDeviceReset()` flushes the output. |
| `helloworld-2.cu` | Prints the block and thread id, with `<<<3,8>>>`. |
| `vecsum-sb.cu` | Vector add with one block of N threads (index = `threadIdx.x`). |
| `vecsum-mb.cu` | Vector add with N blocks of one thread (index = `blockIdx.x`). |
| `vecsum-mb-mb.cu` | 3 blocks × 3 threads, `index = blockIdx.x*blockDim.x + threadIdx.x`. |
| `vecsum-grid.cu` | A 3×3 grid of 3-thread blocks covering 27 elements, `index = bx*blockDim.x*gridDim.y + by*blockDim.x + tx`. |
| `dotproduct.cu` | Grid-stride loop and shared-memory tree reduction. The result is checked against a CPU `dot_product()`. |
| `dotproductTiming.cu` | `dot_shared` (shared-memory reduction per block) against `dot_no_shared` (each thread `atomicAdd`s into a per-block global slot). Timed with `cudaEvent`. With a = b = 1 the expected result is N. |

## Matrix multiplication

| File | What it shows |
|------|---------------|
| `matMul.cu` | 1024×1024 matrices computed four ways: `matMulSerial`, `matMulOpenMP` (`parallel for` over rows), `matMulGPU` (naive, one thread per element) and `matMulGPUShared` (32×32 shared-memory tiles). CPU runs are timed with `omp_get_wtime()` and GPU runs with CUDA events. All results are checked against the serial result. |
| `matMulMultiGPU.cu` | **Multi-GPU with OpenMP.** One host thread per GPU (`omp parallel num_threads(deviceCount)`). Each GPU gets a band of rows of A and C plus a full copy of B, and the bands are copied back into place. GPU contexts are warmed up with `cudaFree(0)` before timing. Compared with serial CPU and single-GPU runs. |
| `matMulStreams.cu` | **Multi-GPU with streams.** The same decomposition, but one host thread queues `cudaMemcpyAsync`, the kernel and the copy-back on a separate stream per GPU, then calls `cudaStreamSynchronize` on each. Host buffers are pinned (`cudaMallocHost`) so the copies really are asynchronous. |

The tiled kernel in the multi-GPU files takes an extra `height` argument (the rows in this GPU's band) and checks bounds on loads and stores.

## Things to notice

- Kernel timing should use CUDA events, because launches return before the kernel finishes. CPU timing should use a wall clock (`omp_get_wtime` or `std::chrono`), not `clock()`.
- Atomics are simpler to write but serialize on contended addresses. Compare the two kernels in `dotproductTiming.cu`.
- For multi-GPU work, the choice of host-side parallelism (OpenMP threads or one thread with streams) changes how work is issued, not what the kernel does.
