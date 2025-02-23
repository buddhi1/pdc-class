CC = nvcc -arch=sm_89	#compiler

SOURCES = info.cu warp.cu mem.cu bankconf.cu
# object file names are same as SOURCES, excluding .cu extension
OBJECTS = $(SOURCES:%.cu=%)

# compile all files in SOURCES 
all: $(SOURCES) $(OBJECTS) 
# compile by file name
# $@ refers to the name of whichever target caused the rule’s recipe to be run
# $< refers to the first prerequisite required to create the output file
%: %.cu
	$(CC) -o $@ $<
# deletes all object files
clean:
	rm $(OBJECTS)
