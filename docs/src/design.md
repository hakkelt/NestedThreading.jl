# Design

## Limits are process-wide

BLAS and FFTW thread counts are process-wide settings of C libraries; there is no way to give
one task a different count from another. NestedThreading therefore keeps every open scope in
one process-wide list. While any scope is open, each library runs at the minimum count of the
open scopes. The counts are saved when the first scope opens and restored when the last one
closes.

Saving and restoring per scope does not work, even under a lock, because the save and the
restore of two tasks interleave:

| | task A | task B | BLAS |
| --- | --- | --- | --- |
| 1 | saves 8, sets 1 | | 1 |
| 2 | | saves **1**, sets 1 | 1 |
| 3 | restores 8 | | 8 |
| 4 | | restores **1** | 1 |

After this, BLAS stays at 1 with no scope open.

!!! warning "While any task is restricted, unrelated concurrent tasks are restricted too"
    A long-running restricted scope slows unrelated work in the same process. The other
    choice, letting the largest request win, oversubscribes the machine, which is what this
    package prevents.

## Some libraries are all or nothing

Polyester and NFFT are turned off whenever the limit is below `capacity()`, not only when it is
1. A limit of 4 inside an 8-thread loop means the machine is already busy, and a nested
Polyester loop would add threads on top.

Polyester can reserve some of its workers instead of all. Reserving only part measures 2.7x
faster at `nthreads() = 8` with 2 outer iterations, and 320x slower at `nthreads() = 48`.
With many threads, a nested parallel loop is very slow however few workers it gets, so the
Polyester guard reserves all of them.

## Guards are decided when a scope opens

A thread count is written every time a scope opens or closes, so it always follows the current
minimum. A guard wraps `f()` and cannot change once `f()` runs. If task A opens an
unrestricted scope and task B then opens a scope at 1, BLAS drops to 1 for both, but A's nested
`@batch` loops still use their workers. This cannot be fixed from outside: Polyester's guard
reserves workers around a call, and another task cannot add that reservation to a call already
running. To avoid it, open the restricting scope before spawning tasks, as the loop macros do.
The opposite case is safe: a scope that opened restricted stays guarded until it closes, even
after the other scope has closed.

## Measure at the thread count you deploy with

Whether budgeting helps depends on how `Threads.nthreads()` compares to the number of cores.
Measured on one 48-core machine (`julia --project=test test/runtests.jl :benchmark`):

| benchmark | `nthreads()` | unbudgeted | budgeted | speedup |
|---|---|---|---|---|
| `nthreads()` × GEMM(384) | 8 | 0.0280 s | 0.0190 s | 1.5x |
| `nthreads()` × GEMM(384) | 48 | 85.998 s | 0.102 s | 846x |
| `nthreads()` × `@batch` over 16384 | 8 | 0.00256 s | 0.00405 s | 0.63x |
| `nthreads()` × `@batch` over 16384 | 48 | 1.103 s | 0.0135 s | 82x |

With threads to spare, nesting costs little and
restricting can be slower. With every core in use, unbudgeted nesting is orders of magnitude
slower. The scopes themselves cost nothing measurable: against a plain `Threads.@threads`
loop, three controls measure 0.99–1.00x.
