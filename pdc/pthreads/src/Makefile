C = gcc
C_FLAGS = -g -Wall -pthread
LD_FLAGS =
LIBS = -lm
SOURCES = 1-cancel.c 5-barrier.c 9-consumer_producer.c 2-join.c 6-cond.c 3-detach.c  7-local_storage.c 4-mutex.c 8-key.c 10-pi.c 11-pi_spaced_1.c 12-pi_spaced_32.c
OBJECTS = $(SOURCES:%.c=%)
# BINS = $(OBJECTS:%.o=%.bin)
DEPENDS = 

all: $(SOURCES) $(OBJECTS)

%: %.c $(DEPENDS) 
	$(CC) -o $@ $< $(C_FLAGS)

clean:
	rm $(OBJECTS)