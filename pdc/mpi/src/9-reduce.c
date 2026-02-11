#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * sum all data from all processes and store the result to process 0
 */
int main (int argc, char *argv[])
{
	int  proc_cnt, proc_id;
	int sum = 0;
	int data = 0;

	/* initialize MPI */
	MPI_Init(&argc, &argv);

	/* get proc id */
	MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

	/* log the sum */
	printf("Before reduction, process %d: the sum is %d\n ", proc_id, sum);

	/* each process generate data for gethering */
	data = proc_id + 1;
	printf("After local processing, process %d: the sum is %d\n ", proc_id, sum);

	/* 
	 * sum all data
	 */
	MPI_Reduce(&data, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	/* log the sum */
	printf("  After reduction, process %d: the sum is %d\n ", proc_id, sum);


	/* cleanup */
	MPI_Finalize();

	return 0;
}
