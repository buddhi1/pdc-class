# CUDA Matrix Multiplication Progression (PDC)

A step-by-step series of matrix-multiply kernels (plus one reduction and one memory benchmark) that matches the "Introduction to CUDA" slides. [`README.txt`](README.txt) gives the compile and run commands for the core examples. This file summarizes **all** the programs. OpenMP-offload versions of the core kernels are in [`../../openmp/src1.5`](../../openmp/src1.5).

## Build and run

```bash
bash run_examples.sh                          # builds and runs the 4 core examples
nvcc -O2 -o MatMul_SharedMemory MatMul_SharedMemory.cu
nvcc -O2 -arch=sm_80 ...                      # optional: target A100 (sm_90 for H100)
```

Inputs are filled with constants (A = 1, and B = 1 or 2.5), so every element of C is known: `n` or `2.5·n`. Most programs print `Verification: PASS/FAIL`.

## Files, in teaching order

| File | Idea | Arguments |
|------|------|-----------|
| `MatMul.cu` / `MatMul_OneBlock.cu` | One block covers the whole matrix, and one thread computes one `C[row][col]` (`threadIdx.x` → column, `threadIdx.y` → row). Limited to n·n ≤ 1024 threads. | `n` |
| `MatMul_OneBlock_Timed.cu` | The same kernel with a warm-up launch and `cudaEvent` timing | `[n]` |
| `MatMul_OneBlock_RowBands.cu` | A 1-D single block in which each thread computes **whole rows** of C (rows `tid`, `tid+T`, …), so n is no longer capped at 32 | `[n] [threads]` |
| `MatMul_OneBlock_RowBands_BT.cu` | The same, but B is transposed on the host first, so both inner-product operands are read row-wise | `[n] [threads]` |
| `MatMul_MultipleBlock.cu` | A 2-D grid of 2-D blocks. Global `row`/`col` come from `blockIdx*blockDim+threadIdx`, with a bounds check for sizes that are not multiples of the block. | `n block` |
| `MatMul_SharedMemory.cu` | **Tiled** multiply. Each block loads one tile of A and one of B into dynamic `extern __shared__` memory per phase, with `__syncthreads()` before and after use. | `n block` |
| `MatMul_SharedMemoryRegisterTiled.cu` | Shared-memory tiling plus **register tiling**. Each thread computes `OUTS` outputs held in registers, and each block covers `TILE × OUTS·TILE` of C. Configure with `-DTILE=16 -DOUTS=2`. | `[n]` |
| `Reduction_OneBlock.cu` | A shared-memory tree reduction in one block (n a power of two, ≤ 1024) | `n` |
| `Coalescing_2D_Copy_Benchmark.cu` | Copies an n×n matrix twice. In the coalesced version `threadIdx.x` walks columns; in the uncoalesced version it walks rows. Prints the slowdown factor. | `[n] [reps]` |

## Helpers

- `checkCuda(err, where)` / `CHECK_CUDA(call)` print the CUDA error string and exit. Kernel launches are checked with `cudaGetLastError()`.
- Timing uses `cudaEventRecord` / `cudaEventElapsedTime` around the kernel only, so host↔device copies are excluded. Run each program a few times, because the first launch includes warm-up costs.
