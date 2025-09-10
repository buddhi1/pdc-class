#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main( int argc, char *argv[] ) {
	int nproc, rank, len;
	char host_name[MPI_MAX_PROCESSOR_NAME];

	double time; 
	MPI_Init (&argc,&argv); /* Initialize MPI */
	
	MPI_Comm_size(MPI_COMM_WORLD,&nproc); /* Get Comm Size*/
	MPI_Comm_rank(MPI_COMM_WORLD,&rank); /* Get rank */
	MPI_Get_processor_name(host_name, &len);
	
	time = MPI_Wtime();

	printf("Hello World from process %d on %s\n", rank, host_name);

	MPI_Barrier(MPI_COMM_WORLD);
	time = MPI_Wtime() - time;

	if(rank==0)
		printf("From process %d execution time: %lf\n", rank, time);

	
	
	MPI_Finalize(); /* Finalize */
	return 0; 
}
