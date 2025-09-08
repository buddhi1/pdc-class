C = gcc
C_FLAGS = -g -Wall -pthread
LD_FLAGS =
LIBS = -lm
SOURCES = pthread_barrier.c pthread_cancel.c pthread_cond.c pthread_create.c pthread_detach.c pthread_key.c pthread_mutex.c consumer_producer.c thread_local_storage.c
OBJECTS = $(SOURCES:%.c=%)
# BINS = $(OBJECTS:%.o=%.bin)
DEPENDS = 

all: $(SOURCES) $(OBJECTS)

%: %.c $(DEPENDS) 
	$(CC) -o $@ $< $(C_FLAGS)

clean:
	rm $(OBJECTS)