CC = nvcc

# 1. -Xcompiler passes flags to GCC (Host compiler)
# 2. -fopenmp enables parallelism
# 3. -O3 turns on high-performance optimization (Critical for Matrix Mul)
CFLAGS = -O3
OMP_FLAGS = -Xcompiler -fopenmp

SOURCES = add.cu dotproduct.cu dotproductTiming.cu helloworld.cu helloworld-2.cu vecsum-grid.cu vecsum-mb.cu vecsum-sb.cu vecsum-mb-mb.cu matMul.cu matMulMultiGPU.cu matMulStreams.cu

# Generate object names from source names
OBJECTS = $(SOURCES:%.cu=%)

# Default target
all: $(OBJECTS)

# ---------------------------------------------------------
# Specific Rules for OpenMP files
# ---------------------------------------------------------

# Compile matMul with OpenMP AND Optimization
matMul: matMul.cu
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

# Compile matMulMultiGPU with OpenMP AND Optimization
matMulMultiGPU: matMulMultiGPU.cu
	$(CC) $(CFLAGS) $(OMP_FLAGS) -o $@ $<

# ---------------------------------------------------------
# Generic Rule for standard CUDA files
# ---------------------------------------------------------
%: %.cu
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(OBJECTS)