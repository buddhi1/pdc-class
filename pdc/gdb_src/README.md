# GDB Debugging Exercises (PDC)

> **Animated walkthrough:** step through every example here, with the matching source lines highlighted, in the [Visual Lab: Debugging with gdb](https://buddhi1.github.io/pdc-class/#/pdc/gdb).

Four small C programs, each with a **deliberate bug**, for practising `gdb`. The comment at the top of each file names the bug. Hide it if you want students to find it themselves.

## Build

```bash
make            # gcc -g; parallel_sum also gets -fopenmp
gdb ./factorial
```

Prebuilt binaries (`divide`, `factorial`, `parallel_sum`, `prefixsum`) are included.

## Files

| File | What it should do | Planted bug | What you see |
|------|-------------------|-------------|--------------|
| `factorial.c` | Compute 5! | The loop runs `i < n` instead of `i <= n` | Prints 24 instead of 120 |
| `divide.c` | Divide two floats | Division by zero is not checked (the check is commented out) | `4 / 0` prints `inf` |
| `prefixsum.c` | In-place prefix sum of 10 ints | Uses `arr[i-2]` instead of `arr[i-1]`, and the loop runs to `i <= n` | Wrong sums, and reads and writes outside the array |
| `parallel_sum.c` | OpenMP sum of 0..N-1 with `reduction(+:sum)` | The loop runs `i <= N` | 15 instead of 10. `_DEBUG_` prints which thread runs each iteration. |

## gdb quick reference

```text
break factorial        # or: break factorial.c:9
run
next / step            # step over / into
print result
display i              # print i after every step
watch result           # stop whenever result changes
info locals
backtrace
info threads           # OpenMP threads (parallel_sum)
thread 2
```
