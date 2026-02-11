CC = mpicc
C_FLAGS = -g -Wall
LD_FLAGS = 
LIBS = 
SOURCES = 10-allreduce.c 15-management.c 5-broadcast.c 11-nonblock_comm.c 1-hello.c 6-gather.c 12-nonblock_comm_2.c  2-hello2.c 7-scatter.c 13-group.c 3-block_comm.c 8-allgather.c 14-custom_type_op.c 4-comb_sendrecv.c 9-reduce.c
OBJECTS = $(SOURCES:%.c=%)
# BINS = $(OBJECTS:%.o=%.bin)
DEPENDS = 

all: $(SOURCES) $(OBJECTS)

%: %.c $(DEPENDS) 
	$(CC) -o $@ $< $(C_FLAGS)

clean:
	rm $(OBJECTS)
