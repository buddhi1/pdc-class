# CUDA Streams, Multi-GPU, Occupancy and Memory (PDC)

Programs adapted from the **NVIDIA CUDA Samples** (BSD license, see the file headers), covering features beyond basic kernels. Comments marked `// *****` are course annotations.

## Build and run

```bash
make                                  # nvcc -I Common/ ...
./simpleMultiCopy
./simpleStreams --sync_method=4       # with no option it only prints help
```

## Files

| File | Topic | What it does |
|------|-------|--------------|
| `simpleStreams.cu` | **Streams and overlap** | Initializes an array on the GPU and copies it back, first without streams and then split across 4 streams so that kernels and copies overlap. Uses pinned host memory (`cudaMallocHost`, or `mmap` + `cudaHostRegister`). Options: `--sync_method=0/1/2/4`, `--use_generic_memory`, `--use_cuda_malloc_host`. |
| `simpleMultiCopy.cu` | **Overlapping two copies with a kernel** | Measures host-to-device, device-to-host and kernel time separately, prints the theoretical speedup limits, then compares a serialized run with a 4-stream pipelined run (`processWithStreams`). |
| `simpleMultiGPU.cu` + `.h` | **Multiple GPUs** | Splits an array across every visible GPU. Each GPU gets its own stream and runs `reduceKernel`, and the CPU combines the partial sums and checks them against a CPU sum. `TGPUplan` (in the `.h`) holds each GPU's buffers, stream and data size. |
| `simpleZeroCopy.cu` | **Zero-copy (mapped) memory** | Allocates host memory with `cudaHostAllocMapped`, gets device pointers with `cudaHostGetDevicePointer`, and runs `vectorAddGPU` directly on host memory with no `cudaMemcpy`. |
| `simpleOccupancy.cu` | **Occupancy API** | Launches `square` with a manual block size (32) and then with `cudaOccupancyMaxPotentialBlockSize`. `reportPotentialOccupancy()` reports active warps ÷ max warps per SM, using `cudaOccupancyMaxActiveBlocksPerMultiprocessor`. |
| `simpleCooperativeGroups.cu` | **Cooperative groups** | `sumReduction(thread_group g, ...)` is a generic reduction that works on any group. It runs on the whole block (`this_thread_block()`) and then on 16-thread tiles (`tiled_partition<16>`). |

## `Common/` helper headers (from the CUDA Samples)

| Header | Main helpers |
|--------|--------------|
| `helper_cuda.h` | `checkCudaErrors(call)`, `getLastCudaError(msg)`, `findCudaDevice`, `gpuDeviceInit`, `gpuGetMaxGflopsDeviceId`, `_ConvertSMVer2Cores` (cores per SM) |
| `helper_functions.h` | Includes the headers below |
| `helper_string.h` | Command-line parsing: `checkCmdLineFlag`, `getCmdLineArgumentInt/String` |
| `helper_timer.h` | `StopWatchInterface`, `sdkCreateTimer/StartTimer/StopTimer/GetTimerValue` |
| `helper_image.h`, `exception.h` | Image and file I/O helpers and exception wrappers (not used directly here) |
