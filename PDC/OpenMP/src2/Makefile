C = gcc
C_FLAGS = -g -Wall -fopenmp
LD_FLAGS =
LIBS = -lm
SOURCES = parallel_directive.c section_directive.c single_directive.c master_directive.c barrier_directive.c critical_directive.c atomic_directive.c private_clause.c shared_clause.c firstprivate_clause.c lastprivate_clause.c reduction_clause.c nowait_clause.c ompgetthreadnum_function.c
OBJECTS = $(SOURCES:%.c=%)
DEPENDS = 

all: $(SOURCES) $(OBJECTS)

%: %.c $(DEPENDS) 
	$(CC) -o $@ $< $(C_FLAGS)

clean:
	rm $(OBJECTS)
