# Restrict a call that is not a loop

## Turn threading off or on

[`with_threads_disabled`](@ref) limits every library to one thread while `f()` runs;
[`with_threads_enabled`](@ref) asks for every thread, and also turns on libraries that are
off by default (NFFT). Both return `f()`'s value.

```julia
with_threads_disabled() do
    @sync for idx in indices
        Threads.@spawn work(idx)
    end
end

parallel ? with_threads_enabled(() -> f(x)) : with_threads_disabled(() -> f(x))
```

An enclosing limit still applies: `with_threads_enabled` inside a saturated loop runs at the
loop's budget. [`with_thread_limit`](@ref) takes an explicit count.

## Restrict only some libraries

Every scope takes `only` (restrict these libraries) or `exclude` (restrict all but these).
Each accepts one name or a collection of names. Libraries the scope does not cover keep their
current count.

```julia
with_threads_disabled(only = (:blas, :mkl)) do
    cg!(x, A, b)
end
```

A name that is not registered is ignored, so you can name a library whose package is not
loaded.

## Run serially, but thread the large calls

A computation that makes many small calls and a few large ones (an iterative solver with an
occasional SVD) can set a soft default with [`with_thread_default`](@ref) and widen the
large calls with [`with_thread_grant`](@ref):

```julia
using LinearAlgebra

with_thread_default(1; only = :blas) do
    for k in 1:iterations
        axpy!(α, x, y)                       # one BLAS thread
        if k % 50 == 0
            with_thread_grant(NestedThreading.capacity(); only = :blas) do
                svd!(copy(A))                # every thread
            end
        end
    end
end
```

A grant never goes past an enclosing [`with_thread_limit`](@ref), so the same code inside a
saturated batch loop stays at one thread.

## Inspect the registry

[`thread_pool_info`](@ref) lists every registered library with its controls and its current,
unrestricted and effective thread counts:

```julia
[p.name => p.current for p in thread_pool_info() if p.has_count]
```

[`NestedThreading.thread_limit`](@ref) returns the hard limit the enclosing scopes put on one
library, or `nothing`.

## Reset the counts

[`NestedThreading.reset_thread_counts!`](@ref) sets every count back to its value when the
library registered, for example after `BLAS.set_num_threads(1)` was called by hand. Open
scopes still apply; the reset changes what the counts return to when the last one closes.

## See what a scope skipped

A scope never throws because of a library. It skips what it cannot do and logs it at the
`Debug` level: an unregistered name in `only`, `exclude` or `thread_limit`, a count below 1
(raised to 1), a `park` hook held back by a partial limit, a guard registered while a scope
is open, and an OpenBLAS without `blas_thread_shutdown_`. To see these messages:

```julia
ENV["JULIA_DEBUG"] = "NestedThreading"
```

A library hook that throws is skipped too. Its first failure is logged as a warning, later
ones at `Debug`; the scope carries on with the other libraries and returns or rethrows what
its body did.
