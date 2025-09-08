#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main( int argc, char *argv[] ) {
	int nproc, rank;
	double time; 
	MPI_Init (&argc,&argv); /* Initialize MPI */
	
	MPI_Comm_size(MPI_COMM_WORLD,&nproc); /* Get Comm Size*/
	MPI_Comm_rank(MPI_COMM_WORLD,&rank); /* Get rank */
	
	time = MPI_Wtime();

	printf("Hello World from process %d\n", rank);

	MPI_Barrier(MPI_COMM_WORLD);
	time = MPI_Wtime() - time;

	if(rank==0)
		printf("From process %d execution time: %lf\n", rank, time);

	
	
	MPI_Finalize(); /* Finalize */
	return 0; 
}
