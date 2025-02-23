CC = nvcc
C_FLAGS = 
LD_FLAGS = 
LIBS = 
SOURCES = matrixMul_shared.cu matrixMulOrg.cu add.cu dotproduct.cu helloworld.cu helloworld-2.cu vecsum-grid.cu  vecsum-mb.cu vecsum-sb.cu 
OBJECTS = $(SOURCES:%.cu=%)
# BINS = $(OBJECTS:%.o=%.bin)
DEPENDS = 

all: $(SOURCES) $(OBJECTS)

%: %.cu $(DEPENDS) 
	$(CC) -o $@ $< $(C_FLAGS)

clean:
	rm $(OBJECTS)
