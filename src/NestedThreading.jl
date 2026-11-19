"""
    NestedThreading

Coordinate a shared CPU thread budget across independently-threaded libraries (BLAS, FFTW,
Polyester, NFFT, …) so that an outer parallel loop does not oversubscribe the machine with
inner threading — Julia's answer to Python's `threadpoolctl`.

The entry points are [`@budgeted_threads`](@ref) / [`@budgeted_batch`](@ref) for loops,
[`@budgeted`](@ref) for a loop construct you want to keep exactly as written, and
[`with_threads_disabled`](@ref) / [`with_threads_enabled`](@ref) for call sites that are not
loops. Libraries register themselves through package extensions; see
[`register_pool!`](@ref) to add one this package does not ship.

Budgeting is one half of the problem. The other is that a library's workers can go on
*occupying* Julia's threads after its own parallel region has ended, so that a
`Threads.@threads` region opened next waits for them instead of running on them — 38x on the
measurement in [`register_pool!`](@ref). The loop macros here handle that for themselves;
[`quiesce_foreign_pools`](@ref) is the same thing for a parallel region written by hand.
"""
module NestedThreading

using Base.Threads: threadpoolsize
using Libdl: Libdl
using LinearAlgebra: BLAS

export @budgeted, @budgeted_threads, @budgeted_batch,
    with_threads_disabled, with_threads_enabled, with_thread_limit,
    with_thread_default, with_thread_grant

@static if VERSION >= v"1.11"
    eval(
        Expr(
            :public,
            :register_pool!, :capacity, :thread_limit, :quiesce_foreign_pools,
            :reset_thread_counts!,
        )
    )
end

include("registry.jl")
include("scopes.jl")
include("macros.jl")

"""
    park_openblas()

Shut OpenBLAS's worker threads down, if OpenBLAS is the BLAS libblastrampoline currently
forwards to; do nothing otherwise. Registered as the `park` hook of the `:blas` pool.

Lowering OpenBLAS's thread count leaves its now-unused workers spinning for
`OPENBLAS_THREAD_TIMEOUT` before they sleep, on the cores the caller's own threads want next.
Measured on an AMD EPYC 7352, 8 Julia threads, Julia 1.13.0, OpenBLAS 0.3.30, 2026-09-23: a
`Threads.@threads` region run right after an 8-thread 1024² `gemm` and a drop to 1 BLAS thread
took 31.0 ms, against 13.2 ms on an idle machine and 14.3 ms after `blas_thread_shutdown_`,
which itself took 0.35 ms. OpenBLAS starts the workers again at its next threaded call, for
about 0.5–1 ms.

The shutdown must not overlap a threaded OpenBLAS call on another task: it tells every worker
to exit and joins it, and a worker in the middle of another caller's work item never sees the
request, so the join would hang. Two rules keep that from happening. `_exit!` runs the
hook under the registry lock, which orders it against every other scope, and skips it while a
hard limit strictly between 1 and [`capacity`](@ref) is open — the limit a
[`@budgeted_threads`](@ref) loop with spare threads per worker opens, whose other workers may be
in such a call. A task that calls BLAS with no scope of its own, at a count a grant raised, is
the one case neither rule covers.

MKL registers no hook: `omp_pause_resource_all` measured a smaller gain there and made the
next threaded call much slower, and `KMP_BLOCKTIME` already bounds how long its workers spin.
"""
function park_openblas()
    for lib in BLAS.get_config().loaded_libs
        handle = Libdl.dlopen(lib.libname; throw_error = false)
        handle === nothing && continue
        shutdown = Libdl.dlsym(handle, :blas_thread_shutdown_; throw_error = false)
        shutdown === nothing || ccall(shutdown, Cint, ())
        Libdl.dlclose(handle)
    end
    return nothing
end

function __init__()
    register_pool!(
        :blas; get = BLAS.get_num_threads, set = BLAS.set_num_threads, park = park_openblas
    )
    return nothing
end

end # module NestedThreading
