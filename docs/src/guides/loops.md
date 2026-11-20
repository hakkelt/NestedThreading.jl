# Budget a parallel loop

## `Threads.@threads`

Write [`@budgeted_threads`](@ref) where you would write `Threads.@threads`. Every registered
library is limited to `max(1, capacity() ÷ length(range))` threads while the loop runs.

To switch the loop off from a flag, pass `threads =`:

```julia
@budgeted_threads threads = parallel for i in eachindex(xs)
    process!(xs[i])
end
```

With `threads = false` the loop runs sequentially, and inner libraries are still limited to
one thread: a caller turns threading off when something else is already running in parallel.

Other arguments before the loop go to `Threads.@threads`:

```julia
@budgeted_threads :static threads = parallel for i in eachindex(xs)
    process!(xs[i])
end
```

## Polyester `@batch`

[`@budgeted_batch`](@ref) does the same for `Polyester.@batch` and passes `@batch` options
through. The calling module needs `import Polyester`.

```julia
import Polyester

@budgeted_batch minbatch = 64 threads = parallel for i in eachindex(xs)
    process!(xs[i])
end
```

## Any other loop macro

[`@budgeted`](@ref) wraps a loop construct and emits it unchanged:

```julia
@budgeted Threads.@threads :static for i in 1:n
    process!(i)
end
```

## Hand-written `@threads` or `@spawn` regions

Polyester's workers keep spinning for a while after a `@batch` loop ends, and a
`Threads.@threads` region opened in that window waits for them: an empty loop on 8 threads
takes 219.9 µs instead of 5.7 µs. The loop macros handle this.
Before a region you write yourself, call [`NestedThreading.quiesce_foreign_pools`](@ref):

```julia
NestedThreading.quiesce_foreign_pools()
Threads.@threads for i in 1:n
    process!(i)
end
```

## In package code

- Change thread counts with a scope, not with `BLAS.set_num_threads` and the like. A count set
  directly stays set after the call and changes the count other tasks run with; a scope
  restores it.
- Do not call [`NestedThreading.reset_thread_counts!`](@ref): it overwrites counts the user
  chose.
