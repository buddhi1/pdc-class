CC = gcc	#compiler

SOURCES = factorial.c prefixsum.c divide.c
# object file names are same as SOURCES, excluding .cu extension
OBJECTS = $(SOURCES:%.c=%)

# compile all files in SOURCES 
all: $(SOURCES) $(OBJECTS) parallel_sum
# compile by file name
# $@ refers to the name of whichever target caused the rule’s recipe to be run
# $< refers to the first prerequisite required to create the output file
%: %.c
	$(CC) -g -o $@ $<

# compile OpenMP file
parallel_sum: parallel_sum.c
	$(CC) -fopenmp -g -o parallel_sum parallel_sum.c
# deletes all object files
clean:
	rm $(OBJECTS)
	rm parallel_sum
