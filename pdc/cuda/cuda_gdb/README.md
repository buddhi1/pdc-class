# CUDA Debugging and Profiling (PDC)

Material for a `cuda-gdb` and Nsight Systems demo. The sources are compiled with host and device debug information (`-g -G`).

## Build

```bash
make            # nvcc -g -G -o <name> <name>.cu
```

`-G` turns off device optimizations so that you can step through kernels. Do not use it for timing.

## Files

| File | Purpose |
|------|---------|
| `debug.cu` | Vector add with 10,000 elements and 128 threads per block. The kernel `addArrays` has an `if (tid < size)` guard for the extra threads. This is the main program for stepping through a kernel. |
| `info.cu` | Prints GPU properties (same as `../src/info.cu`). |
| `matMul.cu` | Serial, OpenMP, naive-GPU and shared-memory GPU matrix multiply (same as `../src/matMul.cu`). |
| `debug`, `info`, `matMul` | Prebuilt debug binaries. |
| `matMul_ex` | A prebuilt `matMul` binary without debug information. |
| `report1.nsys-rep`, `report2.nsys-rep` | Nsight Systems profiles captured from these programs. Open them in `nsys-ui`, or print summaries with `nsys stats report1.nsys-rep`. |

## cuda-gdb quick reference

```text
cuda-gdb ./debug
(cuda-gdb) break addArrays          # stop when the kernel starts
(cuda-gdb) run
(cuda-gdb) info cuda kernels        # running kernels
(cuda-gdb) info cuda threads        # thread positions
(cuda-gdb) cuda block (1,0,0) thread (5,0,0)   # switch focus
(cuda-gdb) print tid
(cuda-gdb) next / step / continue
```

To capture a new profile: `nsys profile -o report3 ./matMul`.
