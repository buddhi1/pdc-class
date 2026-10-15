#!/bin/sh
set -e
CC=${CC:-gcc}
CFLAGS="-O2 -fopenmp"

$CC $CFLAGS MatMul_OneBlock_OpenMP_Target.c -lm -o MatMul_OneBlock_OpenMP_Target
$CC $CFLAGS MatMul_MultipleBlock_OpenMP_Target.c -lm -o MatMul_MultipleBlock_OpenMP_Target
$CC $CFLAGS MatMul_SharedMemory_OpenMP_Target.c -lm -o MatMul_SharedMemory_OpenMP_Target
$CC $CFLAGS Reduction_OneBlock_OpenMP_Target.c -lm -o Reduction_OneBlock_OpenMP_Target

./MatMul_OneBlock_OpenMP_Target 16
./MatMul_MultipleBlock_OpenMP_Target 64 8
./MatMul_SharedMemory_OpenMP_Target 64 8
./Reduction_OneBlock_OpenMP_Target 256
