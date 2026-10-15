# CUDA Basics (PDC)

A first set of CUDA programs: launching kernels, mapping threads and blocks to data, shared-memory reduction, timing, and a CPU-vs-GPU matrix multiply. A slightly different HPML copy lives in [`hpml/cuda/src1`](../../../hpml/cuda/src1).

## Build and run

```bash
make                  # nvcc, one binary per .cu file listed in the Makefile
./vecsum-mb-mb
nvcc add.cu -o add    # add.cu is not in the Makefile
```

## Files

| File | What it shows | Launch configuration |
|------|---------------|---------------------|
| `add.cu` | The smallest possible kernel: adds two ints on the GPU and copies the result back with `cudaMemcpy` | `<<<1,1>>>` |
| `helloworld.cu` | `printf` from device code. `cudaDeviceSynchronize()` is needed so the GPU output appears before the program exits. | `<<<1,10>>>` |
| `helloworld-2.cu` | Prints `blockIdx.x` and `threadIdx.x` from every thread | `<<<5,10>>>` |
| `vecsum-sb.cu` | Vector add, one block. Index = `threadIdx.x`. | `<<<1,N>>>` |
| `vecsum-mb.cu` | Vector add, one thread per block. Index = `blockIdx.x`. | `<<<N,1>>>` |
| `vecsum-mb-mb.cu` | The general pattern: `idx = blockIdx.x*blockDim.x + threadIdx.x`, a ceiling-division grid size, and an `if (idx < n)` bounds check | `<<<ceil(N/T),T>>>` |
| `vecsum-grid.cu` | A 2-D grid of 1-D blocks. The kernel flattens `(blockIdx.x, blockIdx.y)` into a linear block id. | `dim3(3,3)` × 3 threads |
| `dotproduct.cu` | Dot product with a grid-stride loop, then a per-block **shared-memory tree reduction** (`__shared__`, `__syncthreads()`). The CPU sums the per-block partial results. | 32 × 256 |
| `dotproductTiming.cu` | Times the shared-memory kernel (`dot1`) against a global-memory version (`dot_global` + `reduce_global`) using `cudaEvent` | 32 × 256 |
| `info.cu` | Prints device properties: SMs, shared memory per block, registers, warp size, thread limits | n/a |
| `matMul.cu` | Matrix multiply four ways: serial CPU, OpenMP CPU, naive GPU (`matMulGPU`), and shared-memory tiled GPU (`matMulGPUShared`). Results are compared with the serial version. | `WIDTH=32`, `BLOCK_SIZE=32` |

## Things to notice

- The usual host workflow is `cudaMalloc` → `cudaMemcpy(HostToDevice)` → kernel launch → `cudaMemcpy(DeviceToHost)` → `cudaFree`.
- The tree reduction in `dotproduct*.cu` needs `threadsPerBlock` to be a power of two.
- In `matMul.cu`, `BLOCK_SIZE` is also the tile size. The shared kernel assumes `WIDTH` is a multiple of `BLOCK_SIZE`.
- `matMul.cu` times its runs with `clock()`, which measures CPU time, not wall time. The Makefile also does not pass `-Xcompiler -fopenmp`, so the "OpenMP" version runs serially. The HPML copy fixes both.
