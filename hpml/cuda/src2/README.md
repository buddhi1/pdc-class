# CUDA Performance Pitfalls (HPML)

Micro-benchmarks for three GPU performance problems: warp divergence, uncoalesced global memory access, and shared-memory bank conflicts. Each file's header comment has the exact `ncu` (Nsight Compute) command and a recorded profile from a compute-capability 7.5 GPU.

## Build and run

```bash
make
./warp
ncu --section WarpStateStats ./warp
```

## Files

| File | Pitfall | Kernels compared | What to measure |
|------|---------|------------------|-----------------|
| `info.cu` | n/a | n/a | Lists every GPU: compute capability, SMs, memory sizes, warp size, thread limits. Uses a `CUDA_CHECK` macro. |
| `warp.cu` | **Warp divergence** | `divergentKernel` branches on even or odd **thread** id, so both paths run inside every warp. `sortedKernel` branches on **warp** id, so each warp takes one path. `heavyMath()` is a compute-bound loop kept alive with `volatile`. | `ncu --section WarpStateStats`: average active threads per warp is 16 (divergent) vs 32 (sorted). |
| `mem.cu` | **Uncoalesced access** | `coalescedKernel` reads `input[idx]`. `stridedKernel` reads `input[idx*32]`, so neighbouring threads are 128 B apart. | `ncu --metrics l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld`: 4 sectors per request (coalesced) vs 32 (strided). |
| `bankconf.cu` | **Shared-memory bank conflicts** | `noBankConflictKernel` has thread *t* read `smem[t]` (32 different banks). `bankConflictKernel` reads `smem[(t*32) % 1024]`, so the whole warp hits bank 0: a 32-way conflict. | `ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`: 0 vs about 9.7 × 10⁸. |

## Helpers

- `measureExecutionTime(kernel, ...)` (in `mem.cu` and `bankconf.cu`) launches a kernel through a function pointer, times it with CUDA events and prints the time (and, in `mem.cu`, the effective bandwidth).
- `warmup()` is an empty kernel launched first, so CUDA context creation is not counted in the first timing.

Bank rule used in `bankconf.cu`: shared memory has 32 banks of 4 bytes, and `bank = (word index) % 32`. Threads in one warp that hit the same bank at different addresses are served one after another.
