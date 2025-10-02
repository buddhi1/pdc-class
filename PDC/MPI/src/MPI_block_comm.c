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
	MPI_Status status;
	int partner;
	int message;

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

	if(proc_id % 2 == 0){
		/* a process with even id*/
		partner = proc_id + 1;
		/* note that here send first, receive second */
		/* try to reverse their orders and see what may happen*/
		MPI_Send(&proc_id, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);
		MPI_Recv(&message, 1, MPI_INT, partner, 1, MPI_COMM_WORLD,
			 &status);
	}
	else{
		/* an process with odd id */
		partner = proc_id - 1;
		/* note that here receive first, send second */
		/* try to reverse their orders and see what may happen*/
		MPI_Recv(&message, 1, MPI_INT, partner, 1, MPI_COMM_WORLD,
			 &status);
		MPI_Send(&proc_id, 1, MPI_INT, partner, 1, MPI_COMM_WORLD);

	}

	/* print matching information */
	printf("Process %d matched to process %d\n", proc_id, message);

cleanup:
	/* cleanup */
	MPI_Finalize();

	return 0;
}
