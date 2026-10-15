# OpenMP GPU Offload: CUDA Analogues (PDC)

> **Animated walkthrough:** step through every example here, with the matching source lines highlighted, in the [Visual Lab: OpenMP GPU offload](https://buddhi1.github.io/pdc-class/#/pdc/openmp-offload).

OpenMP `target` versions of the CUDA teaching kernels in [`../../cuda/src1.5`](../../cuda/src1.5). Each program mirrors the CUDA version's thread mapping, so students can compare the two programming models side by side. Every program checks its own result and prints `Execution: target device | host fallback` and `Verification: PASS | FAIL`.

[`README.txt`](README.txt) has the Enterprise1-specific validation notes.

## How CUDA maps to OpenMP here

| CUDA | OpenMP in these files |
|------|------------------------|
| thread block | team (`target teams num_teams(...)`) |
| thread in a block | thread in a team (`parallel num_threads(...)`) |
| `__shared__` array | array declared inside the `teams` region (one copy per team; it is **not guaranteed** to sit in GPU shared memory) |
| `__syncthreads()` | `#pragma omp barrier` |
| `cudaMemcpy` to and from the device | `target data map(to: ...) map(from: ...)` |

**Common loop pattern:** `for (tid = omp_get_thread_num(); tid < requested; tid += omp_get_num_threads())`. The runtime may create fewer threads than requested, so each real thread covers several of the "CUDA threads". That keeps every output element computed.

`ran_on_device = !omp_is_initial_device()`, run in a tiny target region, reports whether the code actually ran on the GPU.

## Files

| File | CUDA counterpart | Arguments |
|------|------------------|-----------|
| `MatMul_OneBlock_OpenMP_Target.c` | `MatMul_OneBlock.cu`: one team, one thread per C element (n·n ≤ 1024) | `n` |
| `MatMul_OpenMP_Target.c` | `MatMul.cu`. Identical to the one-block version. | `n` |
| `MatMul_MultipleBlock_OpenMP_Target.c` | `MatMul_MultipleBlock.cu`: a 2-D grid of teams, one thread per C element | `n block` |
| `MatMul_SharedMemory_OpenMP_Target.c` | `MatMul_SharedMemory.cu`: tiled multiply with team-local `As`/`Bs` tiles and barriers (tile ≤ 32) | `n tile` |
| `Reduction_OneBlock_OpenMP_Target.c` | `Reduction_OneBlock.cu`: tree reduction in one team (n a power of two, ≤ 1024) | `n` |

## Build and run

```bash
sh run_gpu.sh            # gcc -fopenmp -foffload=nvptx-none ..., OMP_TARGET_OFFLOAD=MANDATORY
sh run_host_fallback.sh  # same programs built without offload; they run on the CPU
```

`run_gpu.sh` needs a GCC built with NVPTX offload support. The extra flags `-fcf-protection=none -fno-stack-protector` are needed by the GCC 13 NVPTX toolchain (see `README.txt`). Neither script builds `MatMul_OpenMP_Target.c`.
