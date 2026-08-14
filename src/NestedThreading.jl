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
"""
module NestedThreading

using Base.Threads: threadpoolsize
using LinearAlgebra: BLAS

export @budgeted, @budgeted_threads, @budgeted_batch,
    with_restricted_threads, with_full_threads, enable_full_threading

include("registry.jl")
include("scopes.jl")
include("macros.jl")

function __init__()
    return register_counted_pool!(BLAS.get_num_threads, BLAS.set_num_threads; name = :blas)
end

end # module NestedThreading
