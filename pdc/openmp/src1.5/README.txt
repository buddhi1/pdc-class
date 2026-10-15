OpenMP NVIDIA GPU validation (Enterprise1)

These files adapt the original CUDA-like mapping so every requested output is
handled even when an OpenMP runtime creates fewer actual threads per team.
The tiled program keeps As and Bs team-local; OpenMP does not promise these
arrays occupy physical GPU shared memory.

In this directory run: sh run_gpu.sh
All four runs must show Execution: target device and Verification: PASS.

Build flags -fcf-protection=none and -fno-stack-protector are required by
Enterprise1's GCC 13 NVPTX toolchain. They affect these builds only.
