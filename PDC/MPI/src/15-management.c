#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main (int argc, char *argv[])
{
	int  proc_cnt, proc_id, len;
	char host_name[MPI_MAX_PROCESSOR_NAME];

	/* initialize MPI */
	MPI_Init(&argc, &argv);

	/* query execution environment */
	MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
	
	MPI_Get_processor_name(host_name, &len);
	
	printf ("Hello from process %d on %s!\n", proc_id, host_name);
	
	if(proc_id == 0)
		printf("MASTER: Number of MPI tasks is: %d\n", proc_cnt);

	/* cleanup */
	MPI_Finalize();

	return 0;
}
