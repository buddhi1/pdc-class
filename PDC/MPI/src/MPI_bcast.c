#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * Broadcast a user-input secret
 */
int main (int argc, char *argv[])
{
	int  proc_cnt, proc_id;
	int secret = '0';

	/* initialize MPI */
	MPI_Init(&argc, &argv);

	/* get proc id */
	MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

	printf("Before bcast, process %d have secret: %c\n", proc_id, (char)secret);

	if(proc_id == 0){
		printf("Please input the secret letter: \n");
		secret = getchar();
		printf("Main process received secret letter; %c\n", (char)secret);
	}


	/* 
	 * Broadcast the secrect to every process
	 */
	MPI_Bcast(&secret, 1, MPI_INT, 0, MPI_COMM_WORLD);

	printf("After bcast, process %d have secret: %c\n", proc_id, (char)secret);

	/* cleanup */
	MPI_Finalize();

	return 0;
}
