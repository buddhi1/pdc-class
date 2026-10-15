# OpenMP Directives and Clauses, Part 2 (PDC)

> **Animated walkthrough:** step through every example here, with the matching source lines highlighted, in the [Visual Lab: OpenMP directives & clauses](https://buddhi1.github.io/pdc-class/#/pdc/openmp-directives).

One small program per OpenMP directive, clause or runtime function. Each file is self-contained and prints which thread did what, so you can see the semantics in the output.

## Build and run

```bash
make                                  # gcc -g -Wall -fopenmp
OMP_NUM_THREADS=4 ./private_clause
make clean
```

## Directives

| File | Directive | Behaviour to observe |
|------|-----------|---------------------|
| `parallel_directive.c` | `parallel` | Every thread runs the block. |
| `barrier_directive.c` | `barrier` | All "Hello" lines print before any "Goodbye". |
| `master_directive.c` | `master` | Only thread 0 runs the block. There is no implied barrier. (OpenMP 5.1 renames this `masked`.) |
| `single_directive.c` | `single` | Exactly one thread (any thread) runs the block, with an implied barrier at the end. |
| `section_directive.c` | `sections` / `section` | Two independent loops run as two sections, possibly on different threads. |
| `critical_directive.c` | `critical` | Increments `i` and `j` without protection (unstable result), then with protection (result equals the thread count). |
| `atomic_directive.c` | `atomic` | The same experiment with `atomic` on a single update. |
| `task_directive.c` | `task` | One thread (inside `single`) creates 16 tasks, and any thread in the team may run them. `i` and `j` are shared, so a task can print an index that has already changed. Adding `firstprivate(i)` to the task captures the value. |

## Data-sharing clauses

| File | Clause | Behaviour to observe |
|------|--------|---------------------|
| `shared_clause.c` | `shared(i)` | One copy of `i` for all threads, updated inside `critical`. |
| `private_clause.c` | `private(i)` | Each thread gets an **uninitialized** copy. The original `i` (11) is unchanged after the region. |
| `firstprivate_clause.c` | `firstprivate(i)` | Like `private`, but each copy starts at the original value (11). |
| `lastprivate_clause.c` | `lastprivate(j)` | After the loop, `j` holds the value from the sequentially last iteration (3). |
| `reduction_clause.c` | `reduction(+:result)` | Dot product of two 8-element vectors. Expected result: 76. |
| `reduction_clause2.c` | `reduction(+:result)` | Dot product with N = 128, a = 1, b = 2. Expected result: 256. |

## Scheduling, synchronization and runtime functions

| File | Topic | Behaviour to observe |
|------|-------|---------------------|
| `schedule_static.c` | `schedule(static, 4)` | Threads receive contiguous chunks of 4 iterations. The comments describe `dynamic`, `guided`, `runtime` (`OMP_SCHEDULE`) and `auto`. |
| `nowait_clause.c` | `nowait` | Removes the implicit barrier after the first `omp for`, so loop-`j` iterations can start before loop `i` has finished. |
| `ompgetthreadnum_function.c` | `omp_get_thread_num()` | Shows which thread handles which of 16 iterations. |
