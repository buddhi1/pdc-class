C = gcc
C_FLAGS = -g -Wall -fopenmp
LD_FLAGS =
LIBS = -lm
SOURCES = atomic.c critical.c for_directive.c hello_threads.c rc.c barrier.c for_directive2.c hello.c
OBJECTS = $(SOURCES:%.c=%)
DEPENDS = 

all: $(SOURCES) $(OBJECTS)

%: %.c $(DEPENDS) 
	$(CC) -o $@ $< $(C_FLAGS)

clean:
	rm $(OBJECTS)

