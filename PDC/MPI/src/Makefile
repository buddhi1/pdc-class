CC = mpicc
C_FLAGS = -g -Wall
LD_FLAGS = 
LIBS = 
SOURCES = MPI_helloworld.c MPI_helloworld2.c MPI_management.c MPI_block_comm.c MPI_nonblock_comm.c MPI_nonblock_comm_2.c MPI_comb_sendrecv.c MPI_bcast.c MPI_scatter.c MPI_gather.c MPI_allgather.c MPI_reduce.c MPI_allreduce.c MPI_custom_type_op.c MPI_group.c
OBJECTS = $(SOURCES:%.c=%)
# BINS = $(OBJECTS:%.o=%.bin)
DEPENDS = 

all: $(SOURCES) $(OBJECTS)

%: %.c $(DEPENDS) 
	$(CC) -o $@ $< $(C_FLAGS)

clean:
	rm $(OBJECTS)
