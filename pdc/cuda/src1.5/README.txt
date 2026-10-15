CUDA TEACHING CODES FOR "Introduction to CUDA"
================================================

These files intentionally match the simple progression used in the slide deck.
They are not intended to replace optimized libraries such as cuBLAS.

1. MatMul.cu
   - Same simple one-block example as MatMul_OneBlock.cu.
   - Included so the compile command on the introductory compiler slide works.

2. MatMul_OneBlock.cu
   - Slides: one-block matrix multiplication / thread-block limit.
   - One thread computes one C[row][col].
   - Whole n x n matrix is handled by a single n x n block.
   - Current limit: n*n <= 1024 threads.

   Compile:
     nvcc -O2 -o MatMul_OneBlock MatMul_OneBlock.cu

   Run:
     ./MatMul_OneBlock 16
     ./MatMul_OneBlock 32
     ./MatMul_OneBlock 33     # program explains why this is too large

3. MatMul_MultipleBlock.cu
   - Slides: multiple blocks.
   - One thread computes one C[row][col].
   - Global row/column are computed from blockIdx, blockDim, and threadIdx.
   - Includes a bounds check for matrices not divisible by block size.
   - Uses CUDA events for kernel timing.

   Compile:
     nvcc -O2 -o MatMul_MultipleBlock MatMul_MultipleBlock.cu

   Run:
     ./MatMul_MultipleBlock 256 16
     ./MatMul_MultipleBlock 1024 16

4. MatMul_SharedMemory.cu
   - Slides: shared-memory tiled matrix multiplication and coalesced loading.
   - Each block computes one tile of C.
   - Each phase cooperatively loads a tile of A and B into shared memory.
   - __syncthreads() is used after loading and before overwriting the tile.
   - Uses CUDA events for kernel timing.

   Compile:
     nvcc -O2 -o MatMul_SharedMemory MatMul_SharedMemory.cu

   Run:
     ./MatMul_SharedMemory 256 16
     ./MatMul_SharedMemory 1024 16

5. Reduction_OneBlock.cu
   - Slide exercise: reduction with n threads.
   - Simple shared-memory tree reduction in one block.
   - To keep the first example simple, n must be a power of 2 and <= 1024.

   Compile:
     nvcc -O2 -o Reduction_OneBlock Reduction_OneBlock.cu

   Run:
     ./Reduction_OneBlock 256

ARCHITECTURE FLAG
-----------------
The programs themselves are generic.  If desired, add an architecture flag:

  H100 (Hopper):  -arch=sm_90
  A100 (Ampere):  -arch=sm_80

For example:
  nvcc -O2 -arch=sm_90 -o MatMul_MultipleBlock MatMul_MultipleBlock.cu

TIMING
------
MatMul_MultipleBlock, MatMul_SharedMemory, and Reduction_OneBlock use CUDA
Events and print kernel-only elapsed time.  Host-to-device and device-to-host
copies are intentionally outside the timed region so the classroom comparison
focuses on the kernels.

For a useful comparison, run the two matrix kernels with the same arguments:

  ./MatMul_MultipleBlock 1024 16
  ./MatMul_SharedMemory  1024 16

Repeat each command a few times because the first run can include warm-up costs.
Exact times depend strongly on the GPU, CUDA version, clocks, and system load.

CORRECTNESS
-----------
A and B are filled with 1.0, so every element of C should equal n.  Each
program checks its output and prints PASS/FAIL.
