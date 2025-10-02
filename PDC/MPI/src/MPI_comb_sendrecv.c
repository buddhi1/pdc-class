#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * Process matching with MPI_send/recv. An even process p is matched with 
 * process (p+1), i.e., 0 matches 1, 2 matches 3, 4 matches 5 ...
 */
int main (int argc, char *argv[])
{
	int  proc_cnt, proc_id;
	int partner;
	int message = 11;
	MPI_Status stat;

	/* initialize MPI */
	MPI_Init(&argc, &argv);

	/* get proc id */
	MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

	if(proc_id == 0)
		printf("MASTER: Number of MPI tasks is: %d\n", proc_cnt);
	if(proc_cnt % 2 != 0){
		/* must have even number of processes */
		printf("Must have even number of processes\n");
		goto cleanup;
	}

	if(proc_id % 2 == 0)
		/* a process with even id*/
		partner = proc_id + 1;
	else
		/* an process with odd id */
		partner = proc_id - 1;

	/* 
	 * MPI_Sendrecv simplifies the code where both send and receive happen back to back
	 */
	MPI_Sendrecv(&proc_id, 1, MPI_INT, partner, 1, &message, 1, MPI_INT, partner, 
		     1, MPI_COMM_WORLD, &stat);

	
	/* print matching information */
	printf("Process %d matched to process %d\n", proc_id, message);


cleanup:
	/* cleanup */
	MPI_Finalize();

	return 0;
}
