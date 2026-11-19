# NestedThreading.jl

!!! note "Disclaimer"
    This package was written mostly by Claude (Anthropic's coding assistant), under human
    direction and review. The design decisions, benchmark numbers and documentation here
    were produced that way; treat the code as you would any other third-party dependency
    and read it before relying on it.

Coordinate one CPU thread budget across independently-threaded libraries, so an outer
parallel loop does not oversubscribe the machine with inner threading.

## The problem

BLAS, FFTW, Polyester and NFFT each decide independently how many threads to use, and none
of them knows it is being called from inside a `Threads.@threads` loop that already occupies
every core:

```julia
Threads.@threads for i in 1:32          # 8 workers
    mul!(view(Y, :, i), A, view(X, :, i))   # ...each starting 8 BLAS threads
end
```

That is 64 runnable threads on 8 cores. Python solves this with
[`threadpoolctl`](https://github.com/joblib/threadpoolctl); this package is the Julia
equivalent.

## Usage

```julia
using NestedThreading

@budgeted_threads for i in 1:32
    mul!(view(Y, :, i), A, view(X, :, i))
end
```

With `Threads.threadpoolsize() == 8` the loop of 32 items already saturates the machine, so every
registered library is limited to one thread inside the body. A loop of 2 items instead gets
a per-worker budget of 4, for 2 workers × 4 inner threads. The arithmetic is
`max(1, capacity() ÷ length(range))`, evaluated at runtime — the range may have a length that
is only known then.

The budget is applied as a thread limit **around the loop**, once, before the parallel region opens; it is not
re-entered on every iteration.

### Turning the loop off

Many operators carry a `threaded::Bool`. Pass it straight through:

```julia
@budgeted_threads threads = op.threaded for i in eachindex(op.batch_indices)
    ...
end
```

`threads = false` runs a plain sequential loop *and* restricts inner libraries to a single
thread, on the assumption that a caller who switched threading off did so because
concurrency is happening somewhere else.

### Polyester loops

[`@budgeted_batch`](@ref) is the same thing for `Polyester.@batch`. The calling module needs
`import Polyester`, since the expansion emits `Polyester.@batch`.

### Options of the emitted loop

Both macros pass every argument before the loop other than `threads =` to the macro they
emit, so a `Threads.@threads` schedule or `@batch` options combine with the switch:

```julia
@budgeted_threads :static threads = op.threaded for i in eachindex(op.batch_indices)
    ...
end

@budgeted_batch minbatch = 64 threads = op.threaded for i in eachindex(op.batch_indices)
    ...
end
```

### Loops that are not `for` loops

`@sync`/`@spawn` blocks, or a single call gated on a flag, use the function forms:

```julia
with_threads_disabled() do
    @sync for idx in indices
        Threads.@spawn work(idx)
    end
end

op.threaded ? with_threads_enabled(() -> f(x)) : with_threads_disabled(() -> f(x))
```

[`with_threads_enabled`](@ref) actively *enables* libraries that default to off (NFFT), and is
still clamped by any outer restriction, so it is safe to call from inside a batch loop.

[`with_thread_limit`](@ref) takes an explicit count, and [`with_thread_default`](@ref) /
[`with_thread_grant`](@ref) let a computation run serial by default and widen only the calls
worth threading; see [Composition rules](@ref).

### Inspecting the registry

[`thread_pool_info`](@ref) lists every registered library with the controls it has and its
current, unrestricted and effective thread counts:

```julia
[p.name => p.current for p in thread_pool_info() if p.has_count]
```

`thread_pool_info(:blas)` gives the same entry for one library, or `nothing` if no library is
registered under that name.

## Seeing what was skipped

NestedThreading never throws at a call site that uses a scope. When it cannot act on part of a
request, it skips that part. Such cases are an `exclude`/`only` or `thread_limit` name no
library has registered under (yet), a thread count below 1 (raised to 1), a library whose
`park` hook is held back because a partial limit is open around it, a guard registered while a
scope is open, and an OpenBLAS shutdown that found no `blas_thread_shutdown_`. Each one is
reported through Julia's standard logging at the `Debug` level. Enable it for this package
with the `JULIA_DEBUG` environment variable, before or after loading it:

```julia
ENV["JULIA_DEBUG"] = "NestedThreading"
```

## Supported libraries

Registration is automatic via package extensions — have the package loaded and it works:

| Library | Controls | What is used |
| --- | --- | --- |
| BLAS | `get`/`set`, `park` | `BLAS.get_num_threads`/`set_num_threads` via libblastrampoline; `park` shuts OpenBLAS's workers down |
| MKL | `get`/`set` | `MKL.get_num_threads`/`set_num_threads`, MKL's own native functions |
| FFTW | `get`/`set` | `FFTW.get_num_threads`/`set_num_threads` |
| NFFT | `get`/`set` (boolean) | `NFFT._use_threads[]` |
| Polyester, LoopVectorization, Octavian | `guard`, `quiesce` | every PolyesterWeave worker reserved, as `disable_polyester_threads` does; covers `@batch`, `@tturbo` and Octavian's threaded kernels |
| Strided | `get`/`set` | `Strided.get_num_threads`/`set_num_threads`, clamped to `Threads.nthreads()` |

MKL gets its own pool alongside the always-on BLAS one: `BLAS.set_num_threads` is supposed
to forward to MKL through libblastrampoline, but the two do not reliably stay in sync
([MKL.jl#174](https://github.com/JuliaLinearAlgebra/MKL.jl/issues/174)), so a scope
also drives MKL's native thread count directly.

See [Adding a library](@ref) to register one this package does not ship.
