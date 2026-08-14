# NestedThreading.jl

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

With `Threads.nthreads() == 8` the loop of 32 items already saturates the machine, so every
registered library is limited to one thread inside the body. A loop of 2 items instead gets
a budget of 4, for 2 workers × 4 inner threads. The arithmetic is
`max(1, capacity() ÷ length(range))`, evaluated at runtime — the range may have a length that
is only known then.

The budget is applied **around the loop**, once, before the parallel region opens; it is not
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

### Loops that are not `for` loops

`@sync`/`@spawn` blocks, or a single call gated on a flag, use the function forms:

```julia
with_restricted_threads() do
    @sync for idx in indices
        Threads.@spawn work(idx)
    end
end

op.threaded ? with_full_threads(() -> f(x)) : with_restricted_threads(() -> f(x))
```

[`with_full_threads`](@ref) actively *enables* libraries that default to off (NFFT), and is
still clamped by any outer restriction, so it is safe to call from inside a batch loop.

## Supported libraries

Registration is automatic via package extensions — have the package loaded and it works:

| Library | Kind | Control used |
| --- | --- | --- |
| BLAS | counted | `BLAS.get_num_threads`/`set_num_threads` via libblastrampoline |
| MKL | counted | `MKL.get_num_threads`/`set_num_threads`, MKL's own native functions |
| FFTW | counted | `FFTW.get_num_threads`/`set_num_threads` |
| NFFT | counted (boolean) | `NFFT._use_threads[]` |
| Polyester | guarded | `disable_polyester_threads` |

MKL gets its own pool alongside the always-on BLAS one: `BLAS.set_num_threads` is supposed
to forward to MKL through libblastrampoline, but the two do not reliably stay in sync
([MKL.jl#174](https://github.com/JuliaLinearAlgebra/MKL.jl/issues/174)), so a budget scope
also drives MKL's native thread count directly.

See [Adding a library](@ref) to register one this package does not ship.
