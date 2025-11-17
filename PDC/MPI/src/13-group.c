#include "mpi.h"
#include <stdio.h>
#define NPROCS 8

int main(int argc, char *argv[])  {
  int proc_id, proc_cnt, new_id;
  int group1_ids[4]={0,1,2,3}, group2_ids[4]={4,5,6,7};
  int send_buf, recv_buf;
  MPI_Group  orig_group, new_group;   // required variables
  MPI_Comm   new_comm;   // required variable

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_cnt);

  if (proc_cnt != NPROCS) {
    printf("Must specify MP_PROCS= %d. Terminating.\n",NPROCS);
    MPI_Finalize();
    return 0;
  }

  send_buf = proc_id;

  // extract the original group handle
  MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

  //  divide tasks into two distinct groups based upon rank
  if (proc_id < NPROCS/2) {
    MPI_Group_incl(orig_group, NPROCS/2, group1_ids, &new_group);
  }
  else {
    MPI_Group_incl(orig_group, NPROCS/2, group2_ids, &new_group);
  }

  // create new new communicator and then perform collective communications
  MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
  MPI_Allreduce(&send_buf, &recv_buf, 1, MPI_INT, MPI_SUM, new_comm);

  // get rank in new group
  MPI_Comm_rank(new_comm, &new_id);
  printf("rank= %d newrank= %d recvbuf= %d\n",proc_id, new_id, recv_buf);

  MPI_Finalize();

  return 0;
}
