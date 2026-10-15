#!/bin/sh
set -eu
cd "$(dirname "$0")"
CC=${CC:-gcc}
CFLAGS="-O2 -fopenmp -foffload=nvptx-none -fcf-protection=none -fno-stack-protector"

for name in MatMul_OneBlock MatMul_MultipleBlock MatMul_SharedMemory Reduction_OneBlock; do
    "$CC" $CFLAGS "${name}_OpenMP_Target.c" -lm -o "${name}_GPU"
done

export OMP_TARGET_OFFLOAD=MANDATORY
./MatMul_OneBlock_GPU 16
./MatMul_MultipleBlock_GPU 64 8
./MatMul_SharedMemory_GPU 64 8
./Reduction_OneBlock_GPU 256
