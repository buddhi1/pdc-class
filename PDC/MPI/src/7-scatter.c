#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define DATA_COUNT 4

/*
 * scatter data among processes
 */
int main (int argc, char *argv[])
{
	int  proc_cnt, proc_id;
	int data = -1;
	int i;
	int data_to_scatter[DATA_COUNT];

	/* initialize MPI */
	MPI_Init(&argc, &argv);

	/* get proc id */
	MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);


	if(proc_cnt != DATA_COUNT){
	  printf("Must use %d processes\n", DATA_COUNT);
	  goto cleanup;
	}

	printf("Before scatter, process %d have data: %d\n", proc_id, data);

	/*
	 * let process 1 hold the orginal data to scatter
	 */
	if(proc_id == 1){
	  /* generate data to scatter */
	  for(i = 0; i < DATA_COUNT; i++)
	    data_to_scatter[i] = i + 1;
	  
	  /* log the data to scatter */
	  printf("In process %d: the data to scatter are: ", proc_id);
	  for(i = 0; i < DATA_COUNT; i++)
	    printf("%d ", data_to_scatter[i]);
	  printf("\n");
	}


	/* 
	 * scatter the data to every process
	 */
	MPI_Scatter(data_to_scatter, 1, MPI_INT, &data, 1, MPI_INT, 1, MPI_COMM_WORLD);

	printf("After scatter, process %d have data: %d\n", proc_id, data);

 cleanup:
	/* cleanup */
	MPI_Finalize();

	return 0;
}
