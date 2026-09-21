"""
    NestedThreading

Coordinate a shared CPU thread budget across independently-threaded libraries (BLAS, FFTW,
Polyester, NFFT, …) so that an outer parallel loop does not oversubscribe the machine with
inner threading — Julia's answer to Python's `threadpoolctl`.

The entry points are [`@budgeted_threads`](@ref) / [`@budgeted_batch`](@ref) for loops,
[`@budgeted`](@ref) for a loop construct you want to keep exactly as written, and
[`with_restricted_threads`](@ref) / [`with_full_threads`](@ref) for call sites that are not
loops. Libraries register themselves through package extensions; see
[`register_counted_pool!`](@ref) to add one this package does not ship.

Budgeting is one half of the problem. The other is that a library's workers can go on
*occupying* Julia's threads after its own parallel region has ended, so that a
`Threads.@threads` region opened next waits for them instead of running on them — 38x on the
measurement in [`QuiescePool`](@ref). The loop macros here handle that for themselves;
[`quiesce_foreign_pools`](@ref) is the same thing for a parallel region written by hand.
"""
module NestedThreading

using Base.Threads: threadpoolsize
using LinearAlgebra: BLAS

export @budgeted, @budgeted_threads, @budgeted_batch,
    with_restricted_threads, with_full_threads, enable_full_threading,
    quiesce_foreign_pools

include("registry.jl")
include("scopes.jl")
include("macros.jl")

function __init__()
    return register_counted_pool!(BLAS.get_num_threads, BLAS.set_num_threads; name = :blas)
end

end # module NestedThreading
