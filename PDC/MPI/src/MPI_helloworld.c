#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main( int argc, char *argv[] ) {
	int nproc, rank; 
	MPI_Init (&argc,&argv); /* Initialize MPI */
	
	MPI_Comm_size(MPI_COMM_WORLD,&nproc); /* Get Comm Size*/
	MPI_Comm_rank(MPI_COMM_WORLD,&rank); /* Get rank */
	
	printf("Hello World from process %d\n", rank);

	MPI_Finalize(); /* Finalize */
	return 0; 
}
