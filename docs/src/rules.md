# Composition rules

## Budget of a loop macro

`max(1, capacity() ÷ length(range))`, computed when the loop starts, applied as a
[`with_thread_limit`](@ref) around the whole loop. [`@budgeted_batch`](@ref) excludes
`:polyester` so that its own `@batch` loop is not turned off.

## How scopes combine

| scope | kind | combines with the same kind by |
| --- | --- | --- |
| [`with_thread_limit`](@ref) | hard limit | minimum |
| [`with_thread_default`](@ref) | soft default | minimum |
| [`with_thread_grant`](@ref) | grant | maximum |

For each library, from the scopes that cover it: the highest grant applies if there is one,
capped at the library's count with no scope open; otherwise the lowest soft default; otherwise
the lowest hard limit. The result is then capped at the lowest hard limit. The scopes counted
are those open anywhere in the process.

```jldoctest
julia> using NestedThreading, LinearAlgebra

julia> with_thread_limit(1; only = :blas) do
           with_thread_limit(8; only = :blas) do
               BLAS.get_num_threads()
           end
       end
1
```

A limit that cannot lower anything below what the enclosing limits already set runs its body
without opening a scope. "Enclosing" means opened on the same task, or on a task it was
spawned from while the scope was open.

!!! note "Julia 1.10"
    Finding enclosing scopes needs `Base.ScopedValues` (Julia 1.11). On 1.10 every nested
    limit opens a scope, with the same result, and `thread_limit` returns `nothing`.

## All-or-nothing libraries

A library with only a guard (Polyester), or a boolean switch (NFFT), is off whenever the
applied limit is below [`NestedThreading.capacity`](@ref). A guard is entered or not when the
scope opens, from the limit at that moment; see [Design](@ref).

## A limit of `capacity()` raises counts

A scope sets every count it covers to its limit, also upward. [`with_threads_enabled`](@ref),
or a loop with one iteration, sets libraries to `capacity()` even if a lower count was set by
hand; the hand-set count comes back when the scope closes.

```jldoctest
julia> using NestedThreading, LinearAlgebra

julia> n0 = BLAS.get_num_threads(); BLAS.set_num_threads(1)

julia> with_threads_enabled(BLAS.get_num_threads) == NestedThreading.capacity()
true

julia> BLAS.get_num_threads()
1

julia> BLAS.set_num_threads(n0)
```

## Parking

When a grant closes and lowers a library's count, that library's `park` hook runs. The
`:blas` hook shuts OpenBLAS's worker threads down, which would otherwise keep spinning for
about 0.1 s.

## Example

With `capacity() == 8`:

| loop | budget | BLAS/FFTW inside | Polyester/NFFT inside |
| --- | --- | --- | --- |
| `@budgeted_threads for i in 1:32` | 1 | 1 | off |
| `@budgeted_threads for i in 1:2` | 4 | 4 | off |
| `@budgeted_threads for i in 1:1` | 8 | 8 | on |
| `@budgeted_threads threads = false for i in 1:2` | 1 | 1 | off |
