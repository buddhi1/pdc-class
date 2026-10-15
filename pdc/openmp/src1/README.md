# OpenMP Basics, Part 1 (PDC)

> **Animated walkthrough:** step through every example here, with the matching source lines highlighted, in the [Visual Lab: OpenMP basics](https://buddhi1.github.io/pdc-class/#/pdc/openmp-basics).

The first OpenMP examples: parallel regions, thread ids, barriers, race conditions and their fixes, and loop work-sharing.

## Build and run

```bash
make                         # gcc -g -Wall -fopenmp
OMP_NUM_THREADS=4 ./hello
make clean
```

## Files

| File | What it shows |
|------|---------------|
| `hello.c` | `#pragma omp parallel`. Every thread runs the block, so each line prints once per thread in a nondeterministic order. |
| `hello_threads.c` | Same, with `omp_get_thread_num()` printing which thread is speaking. |
| `barrier.c` | `private(id)` plus `#pragma omp barrier`. All "Hello" lines print before any "Goodbye". |
| `rc.c` | **Race condition.** An unprotected `sum = sum + 1` in a parallel region. The final sum can be less than the thread count. |
| `critical.c` | Fixes `rc.c` with `#pragma omp critical` around both the update and the `printf`. |
| `atomic.c` | Fixes `rc.c` with `#pragma omp atomic`. Only the single update is protected: the `printf` that follows reads `sum` without protection, so printed "current" values may repeat, but the final sum is correct. |
| `for_directive.c` | `parallel` plus `omp for` splits 10 iterations among threads. Timed with `omp_get_wtime()`. |
| `for_directive2.c` | `parallel for` on a nested loop. The outer index `i` is private automatically, but the inner index `j` is declared outside and is therefore **shared**, which causes a race. The commented-out `private(j)` is the fix to discuss. |

## Takeaways

- Variables declared before a parallel region are shared by default. Variables declared inside it, and the loop variable of an `omp for`, are private.
- `critical` serializes a whole block. `atomic` protects one memory update and is cheaper.
