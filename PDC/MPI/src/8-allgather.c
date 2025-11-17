#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DATA_COUNT 4

/*
 * gather data into one process (proc 2 here)
 */
int main (int argc, char *argv[])
{
	int  proc_cnt, proc_id;
	int data = -1;
	int i;
	int data_gathered[DATA_COUNT] = {0};

	/* initialize MPI */
	MPI_Init(&argc, &argv);

	/* get proc id */
	MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

	if(proc_cnt != DATA_COUNT){
	  printf("Must use %d processes\n", DATA_COUNT);
	  goto cleanup;
	}

	/* log the gathered data */
	printf("Before gathering, process %d: the array have: ", proc_id);
	for(i = 0; i < DATA_COUNT; i++)
	  printf("%d ", data_gathered[i]);
	printf("\n");

	/* each process generate data for gethering */
	data = proc_id + 1;

	/* 
	 * gather the data 
	 */
	MPI_Allgather(&data, 1, MPI_INT, data_gathered, 1, MPI_INT, MPI_COMM_WORLD);

	/* log the gathered data */
	printf("After gathering, process %d: the array have: ", proc_id);
	for(i = 0; i < DATA_COUNT; i++)
	  printf("%d ", data_gathered[i]);
	printf("\n");

 cleanup:
	/* cleanup */
	MPI_Finalize();

	return 0;
}
