# NestedThreading.jl

[![Tests](https://github.com/hakkelt/NestedThreading.jl/actions/workflows/tests.yml/badge.svg)](https://github.com/hakkelt/NestedThreading.jl/actions/workflows/tests.yml)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://hakkelt.github.io/NestedThreading.jl/dev/)
[![codecov](https://codecov.io/gh/hakkelt/NestedThreading.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/hakkelt/NestedThreading.jl)

> **Disclaimer:** this package was written mostly by Claude (Anthropic's coding assistant),
> under human direction and review. The design decisions, benchmark numbers and
> documentation in this repository were produced that way; treat the code as you would any
> other third-party dependency and read it before relying on it.

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

The budget is `max(1, Threads.threadpoolsize() ÷ length(range))`, computed at runtime, and
applied around the loop — once per loop, not once per iteration.

## API

| Function / macro | Purpose |
| --- | --- |
| `@budgeted_threads [threads=…] for …` | `Threads.@threads` with an automatic inner budget and an on/off switch |
| `@budgeted_batch [threads=…] for …` | the same for Polyester's `@batch` |
| `@budgeted <loop-macro> for …` | budget a loop construct you want emitted exactly as written |
| `with_restricted_threads(f)` | limit everything to one thread for the duration of `f()` — for call sites that are not loops |
| `with_full_threads(f)` | actively request full threading (still clamped by any outer restriction) |
| `enable_full_threading()` | process-wide escape hatch: set the baseline back to full throttle |

Public but not exported: `NestedThreading.with_thread_budget(f, n)`,
`register_counted_pool!`, `register_guarded_pool!`, `CountedPool`, `GuardedPool`,
`capacity`, `budget_for`.

Every scoping function (`with_thread_budget`, `with_restricted_threads`, `with_full_threads`)
also takes `exclude` and `only` keywords to narrow which pools a restriction applies to — a
pool a restriction does not apply to keeps whatever value it already has, not the value the
scope requested:

```julia
# Restrict only BLAS/MKL to one thread, e.g. around a Krylov solve with BLAS-1 inside;
# FFTW and everything else are left exactly as they are.
with_restricted_threads(only = (:blas, :mkl)) do
    cg!(x, A, b)
end
```

`exclude` is a denylist (a name that is not currently registered is simply ignored, not an
error — a caller may exclude a pool whose package is not loaded in this session); `only` is
an allowlist and is the way to say "restrict exactly these pools, leave the rest alone"
without having to enumerate every other pool.

## Supported libraries

Registration happens automatically through package extensions — just have the package
loaded:

| Library | Kind | Notes |
| --- | --- | --- |
| BLAS | counted | always registered; `LinearAlgebra.BLAS.get_num_threads`/`set_num_threads` via libblastrampoline |
| MKL | counted | `MKL.get_num_threads`/`set_num_threads`, MKL's own native functions |
| FFTW | counted | `FFTW.get_num_threads`/`set_num_threads` |
| NFFT | counted (boolean) | `NFFT._use_threads[]`, on only when nothing is restricted |
| Polyester | guarded | `disable_polyester_threads` |

MKL gets its own pool alongside the always-on BLAS one, rather than instead of it:
`BLAS.set_num_threads` goes through libblastrampoline, which is supposed to forward to
MKL, but the two do not reliably stay in sync — setting one does not always update what
the other reports back
([MKL.jl#174](https://github.com/JuliaLinearAlgebra/MKL.jl/issues/174)). Registering both
means every budget scope drives MKL's native thread count directly instead of trusting the
forward.

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

Polyester and NFFT are switched fully off whenever the budget is below `capacity()`, not just
when it is 1 — a budget of 4 inside an 8-thread outer loop must not leave a nested Polyester
loop free to spawn its own workers. Polyester can in fact be limited partially; doing so was
measured 320x worse in the saturated regime and is deliberately not done.

## Tests

```sh
julia --project=test test/runtests.jl              # everything except :benchmark
julia --project=test test/runtests.jl :macros      # one tag
julia --project=test test/runtests.jl :benchmark   # local-only timing comparisons
```

Test items are tagged `:registry`, `:macros`, `:extensions`, `:jet`, `:benchmark`. The
benchmark items use BenchmarkTools with paired measurements and are excluded from the
default run.

CI runs the suite at 4 threads and, in a job of its own, at 1 — `capacity() == 1` collapses
"restricted" and "full throttle" onto the same budget and is a real code path for every
all-or-nothing pool. Worth running locally too: `julia -t 1 --project=test test/runtests.jl`.

**Run them at the thread count you actually deploy with.** Whether budgeting helps depends
on how `nthreads()` compares to the core count, and not by a small factor: on a 48-core
machine, the same nested-GEMM benchmark gains 1.5x at `-t 8` and **846x** at `-t 48`, and the
nested-Polyester one *loses* 1.6x at `-t 8` while gaining **82x** at `-t 48`.
The docs have the full table. The suite also asserts that budget scopes themselves are free
(0.99-1.00x against a plain `Threads.@threads` loop on three separate controls).

## License

MIT — see [LICENSE.md](LICENSE.md).
