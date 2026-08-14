# NestedThreading.jl

Coordinate one CPU thread budget across independently-threaded libraries, so an outer
parallel loop does not oversubscribe the machine with inner threading.

Julia has no general equivalent of Python's [`threadpoolctl`](https://github.com/joblib/threadpoolctl):
BLAS, FFTW, Polyester and NFFT each decide on their own how many threads to use, and none of
them knows it is being called from inside a `Threads.@threads` loop that already occupies
every core. The usual result is `nthreads()²` runnable threads and a slowdown.

```julia
using NestedThreading

# 32 batch items on 8 threads: the outer loop already saturates the machine,
# so every inner library is limited to 1 thread for the duration.
@budgeted_threads for i in 1:32
    mul!(view(Y, :, i), A, view(X, :, i))
end

# 2 batch items on 8 threads: 2 workers × 4 inner BLAS threads.
@budgeted_threads for i in 1:2
    mul!(view(Y, :, i), A, view(X, :, i))
end
```

The budget is `max(1, Threads.nthreads() ÷ length(range))`, computed at runtime, and applied
around the loop — once per loop, not once per iteration.

## API

| Function / macro | Purpose |
| --- | --- |
| `@budgeted_threads [threads=…] for …` (`@ntt`) | `Threads.@threads` with an automatic inner budget and an on/off switch |
| `@budgeted_batch [threads=…] for …` (`@ntb`) | the same for Polyester's `@batch` |
| `@budgeted <loop-macro> for …` (`@nt`) | budget a loop construct you want emitted exactly as written |
| `with_restricted_threads(f)` | limit everything to one thread for the duration of `f()` — for call sites that are not loops |
| `with_full_threads(f)` | actively request full threading (still clamped by any outer restriction) |
| `enable_full_threading()` | process-wide escape hatch: set the baseline back to full throttle |

Public but not exported: `NestedThreading.with_thread_budget(f, n)`,
`register_counted_pool!`, `register_guarded_pool!`, `CountedPool`, `GuardedPool`,
`capacity`, `budget_for`.

## Supported libraries

Registration happens automatically through package extensions — just have the package
loaded:

| Library | Kind | Notes |
| --- | --- | --- |
| BLAS | counted | always registered; covers MKL, OpenBLAS and any other libblastrampoline backend |
| FFTW | counted | `FFTW.get_num_threads`/`set_num_threads` |
| NFFT | counted (boolean) | `NFFT._use_threads[]`, on only when nothing is restricted |
| Polyester | guarded | `disable_polyester_threads` |

MKL deliberately has no extension: `LinearAlgebra.BLAS.set_num_threads` goes through
libblastrampoline, which forwards to every loaded backend including MKL, so a second pool
would only double-book the same knob.

To add a library this package does not ship, call `register_counted_pool!` from your own
`__init__`:

```julia
NestedThreading.register_counted_pool!(MyLib.get_threads, MyLib.set_threads; name = :mylib)
```

## How the budget composes

Budgets are refcounted process-wide. The applied value is the **minimum over every scope
currently active anywhere in the process**, the counts are snapshotted when the first scope
opens, and restored when the last one closes.

That gives two properties that a naive save/restore (even a locked one) does not:

* **Nesting only ever narrows.** An inner `with_thread_budget(f, 8)` inside an outer
  restriction to 1 still runs at 1 — it cannot widen a budget somebody else is relying on.
* **Concurrent scopes cannot corrupt each other.** With per-scope save/restore, two tasks
  interleave as *A saves 8 → B saves 1 → A restores 8 → B restores 1* and leave the process
  permanently throttled. Here the snapshot is taken once and restored once.

The cost is deliberate conservatism: **while any task is restricted, unrelated concurrent
tasks are restricted too.** BLAS and FFTW thread counts are process-global with no per-task
scoping, so there is no version of this that is both safe and per-task; this package chooses
the direction that never oversubscribes.

Libraries with no partial-count API (Polyester, NFFT) are switched fully off whenever the
budget is below `capacity()`, not just when it is 1 — a budget of 4 inside an 8-thread outer
loop must not leave a nested Polyester loop free to spawn its own workers.

## License

MIT — see [LICENSE.md](LICENSE.md).
