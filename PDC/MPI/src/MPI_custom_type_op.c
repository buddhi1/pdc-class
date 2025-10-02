#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * a simple program where each process produce one item with specific value and weight.
 * After the produce, the total value and weight of all items are computed and returned
 * to every process.
 */
struct item{
  int id;
  double weight; // 
  float value; // the actual value of the item
};

/* create a MPI data type for struct item */
MPI_Datatype create_mpi_item_type()
{
	int ret;
	MPI_Datatype new_type;

	int count = 3; // three blocks, 1st block has two ints (id and cnt), 2nd block has 1 
                       // double (weight) and 3rd block has one float (value)
	int block_length[3] = {1, 1, 1}; // 3 block each has 1, 1, and 1 variables
	MPI_Aint offsets[3] = {0}; //array with beginning offset of each block;
	MPI_Datatype types[3] = {MPI_INT, MPI_DOUBLE, MPI_FLOAT}; // old types of each block are
                                                                  // int, double, float
	MPI_Aint int_size, double_size;

	/*
	 * get the offsets for each block
	 */
	offsets[0] = 0; // block 1 starts from the beginning
	MPI_Type_extent(MPI_INT, &int_size); // get the size of MPI_INT
	offsets[1] = int_size * 2; // block 2 starts after two ints
	MPI_Type_extent(MPI_DOUBLE, &double_size); // get the size MPI_DOUBLE
	offsets[2] = int_size * 2 + double_size; // block 3 starts after two MPI_INTs and one 
	                                         // MPI_DOUBLE
	

	ret = MPI_Type_struct(count, block_length, offsets, types, &new_type);
	if(ret != MPI_SUCCESS)
		printf("Type construct error: %d\n", ret);
	MPI_Type_commit(&new_type);
	if(ret != MPI_SUCCESS)
		printf("Type commit error: %d\n", ret);

	return new_type;
}


/* MPI "sum" reduction for struct item */
void item_sum(void *vin, void *vinout, int *len, MPI_Datatype *dptr)
{
	int i;
	struct item tmp;
	struct item *in = vin;
	struct item *inout = vinout;


	for(i = 0; i < *len; i++){
	  /* sum weight and value */
	  tmp.weight = in->weight + inout->weight;
	  tmp.value = in->value + inout->value;
	  
	  /* set the sum to be the out value */
	  *inout = tmp;
	
	  /* process next element */
	  in++;
	  inout++;
	}

	return;
}


/*
 * sum all data from all processes and store the result to process 0
 */
int main (int argc, char *argv[])
{
	int  proc_cnt, proc_id;
	MPI_Datatype mpi_item_type;
	MPI_Op mpi_item_sum;
	struct item product;
	struct item sum;
	int ret;

	/* initialize MPI */
	MPI_Init(&argc, &argv);

	/* get proc id */
	MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);
	MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);

	/* create custom MPI type for struct MPI_OP */
	mpi_item_type = create_mpi_item_type();
	/* create the custom item sum operation */
	ret = MPI_Op_create(item_sum, 0, &mpi_item_sum);
	if(ret != MPI_SUCCESS)
	  printf("Failed to create item  sum operation\n");

	/* randomly generate some items */
	product.weight = proc_id * 10;
	product.value = proc_id * 100;
	product.id = proc_id;


	/* 
	 * sum all data
	 */
	MPI_Allreduce(&product, &sum, 1, mpi_item_type, mpi_item_sum, MPI_COMM_WORLD);

	/* log the sum */
	printf("After reduction, process %d: total weight is %lf, total value is %f\n", 
	       proc_id, sum.weight, sum.value);


	/* cleanup */
	MPI_Finalize();

	return 0;
}
