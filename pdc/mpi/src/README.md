# MPI Examples (PDC)

Numbered MPI programs that move from "hello world" through point-to-point messaging, collectives, non-blocking communication, communicators, and custom datatypes and operations.

## Build and run

```bash
make                          # mpicc -g -Wall, one binary per .c file
mpirun -np 4 ./1-hello
make clean
```

Every program follows the same skeleton: `MPI_Init` → `MPI_Comm_size` / `MPI_Comm_rank` → work → `MPI_Finalize`.

## Files

| File | Concept | Key calls | `-np` |
|------|---------|-----------|-------|
| `1-hello.c` | Rank, size and host name | `MPI_Get_processor_name` | any |
| `2-hello2.c` | Timing a parallel section | `MPI_Wtime`, `MPI_Barrier` | any |
| `3-block_comm.c` | Blocking send/receive. Even ranks pair with rank + 1. Even ranks send first and odd ranks receive first, which avoids deadlock. Try reversing the order. | `MPI_Send`, `MPI_Recv` | even |
| `4-comb_sendrecv.c` | The same exchange in one call, with no deadlock risk | `MPI_Sendrecv` | even |
| `5-broadcast.c` | Rank 0 reads a character from stdin and broadcasts it to all ranks | `MPI_Bcast` | any |
| `6-gather.c` | Every rank sends `rank+1`; rank **2** gathers them | `MPI_Gather` | 4 |
| `7-scatter.c` | Rank **1** scatters `{1,2,3,4}` across ranks | `MPI_Scatter` | 4 |
| `8-allgather.c` | Every rank ends up with the gathered array | `MPI_Allgather` | 4 |
| `9-reduce.c` | Sum of `rank+1`, delivered to rank 0 only | `MPI_Reduce(MPI_SUM)` | any |
| `10-allreduce.c` | The same sum, delivered to every rank | `MPI_Allreduce` | any |
| `11-nonblock_comm.c` | Non-blocking pair exchange. The message printed before the wait may still hold its old value (11). | `MPI_Isend`, `MPI_Irecv`, `MPI_Waitall` | even |
| `12-nonblock_comm_2.c` | With non-blocking calls, the send/receive order no longer matters, so one code path serves both even and odd ranks | same | even |
| `13-group.c` | Splits 8 ranks into two groups, builds a communicator for each, and runs `Allreduce` inside each group | `MPI_Comm_group`, `MPI_Group_incl`, `MPI_Comm_create` | **8** |
| `14-custom_type_op.c` | An MPI datatype for a C `struct item`, plus a user-defined sum operator | `MPI_Get_address`, `MPI_Type_struct`, `MPI_Type_commit`, `MPI_Op_create` | any |
| `15-management.c` | Same as `1-hello.c` (environment query) | | any |

## Notes

- **`14-custom_type_op.c`:** `create_mpi_item_type()` computes field offsets relative to the struct base. `item_sum()` has the signature MPI requires for a user op (`void f(void *in, void *inout, int *len, MPI_Datatype *)`) and sums only `weight` and `value`.
- `MPI_Type_struct` was removed in MPI-3. Recent Open MPI or MPICH builds may reject it. The modern replacement is `MPI_Type_create_struct`, which takes the same arguments.
- Programs that need a specific process count print a message and exit cleanly when run with the wrong `-np`.
