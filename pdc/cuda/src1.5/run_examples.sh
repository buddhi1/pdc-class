#!/usr/bin/env bash
set -e

NVCC=${NVCC:-nvcc}
FLAGS=${FLAGS:--O2}

$NVCC $FLAGS -o MatMul_OneBlock MatMul_OneBlock.cu
$NVCC $FLAGS -o MatMul_MultipleBlock MatMul_MultipleBlock.cu
$NVCC $FLAGS -o MatMul_SharedMemory MatMul_SharedMemory.cu
$NVCC $FLAGS -o Reduction_OneBlock Reduction_OneBlock.cu

echo
echo "=== One block ==="
./MatMul_OneBlock 16
./MatMul_OneBlock 32

echo
echo "=== Multiple blocks ==="
./MatMul_MultipleBlock 256 16

echo
echo "=== Shared-memory tiling ==="
./MatMul_SharedMemory 256 16

echo
echo "=== Reduction ==="
./Reduction_OneBlock 256
