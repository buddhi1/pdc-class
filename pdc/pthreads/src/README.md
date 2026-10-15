# Pthreads Examples (PDC)

Small POSIX-threads programs that introduce thread creation, lifecycle, synchronization, thread-local data, and a Monte Carlo π case study on false sharing.

## Build and run

```bash
make            # gcc -g -Wall -pthread, one binary per .c file
./2-join
make clean
```

Most examples use `TC = 4` threads. Each thread receives its id through a pointer to an array element (`&idx[i]`). It does not get `&i`, because `i` keeps changing in the creating loop.

## Files

| File | What it shows | Key calls |
|------|---------------|-----------|
| `1-cancel.c` | Creates 4 threads. `main` sleeps 5 s and then cancels them all. | `pthread_create`, `pthread_cancel` |
| `2-join.c` | Same threads, but `main` waits for each to finish. Compare with `1-cancel`. | `pthread_join` |
| `3-detach.c` | Threads block on `getchar()`. `main` detaches them, so the `pthread_join` calls that follow fail at once and `main` exits. | `pthread_detach` |
| `4-mutex.c` | A `struct thread_data` carries id, a pointer to a shared counter and a pointer to a mutex. Each thread increments the counter inside a lock. The final value is 14. | `pthread_mutex_init/lock/unlock/destroy` |
| `5-barrier.c` | Every thread prints "first part" before any thread prints "second part". | `pthread_barrier_init/wait/destroy` |
| `6-cond.c` | Interactive. Phase 1 wakes one waiting thread per Enter key (`signal`). A barrier sized `TC+1` (includes `main`) separates the phases. Phase 2 wakes all threads at once (`broadcast`). | `pthread_cond_wait/signal/broadcast` |
| `7-local_storage.c` | A `__thread int` (GCC thread-local storage). Each thread sees its own copy of a "global". | `__thread` |
| `8-key.c` | The POSIX way to keep per-thread data under one shared key. | `pthread_key_create`, `pthread_setspecific`, `pthread_getspecific` |
| `9-consumer_producer.c` | 4 producers and 4 consumers share a `products` counter. Consumers wait in a `while (products == 0)` loop, which protects against spurious wake-ups. | mutex and condition variable |
| `10-pi.c` | Monte Carlo π (Grama et al., ch. 7). Each thread counts hits in a **local** variable and writes it once to `hits[i]`. The program reruns with 1 thread and prints the speedup. | `rand_r` (thread-safe RNG) |
| `11-pi_spaced_1.c` | Same, but each thread increments `hits[i]` directly in the loop. Neighbouring counters share a cache line, which causes **false sharing** and a slowdown. | |
| `12-pi_spaced_32.c` | Pads each counter to `hits[i][32]` (128 bytes), so every thread writes to its own cache line. False sharing goes away. | |

## Notes

- Programs 10–12 read the number of sample points and threads from stdin. Try the same inputs on all three (for example 100000000 points and 8 threads) and compare the times.
- The "sequential" baseline in 10–12 runs the same code with 1 thread, so it still pays thread-creation overhead (the source comments say so).
- Programs 10–12 declare `main()` with no return type, so `-Wall` prints warnings. They still compile.
