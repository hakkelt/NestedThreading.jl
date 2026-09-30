module NestedThreadingPolyesterExt

import NestedThreading
import Polyester

# Polyester has no thread-count setter, so this guard switches it off for the duration of any
# restricted scope. It does that the way `disable_polyester_threads` does, by reserving every
# PolyesterWeave worker slot, so nested `@batch` loops find none free and run serially. Reservation is
# inherently nesting- and concurrency-safe (an atomic take from a shared mask), which is why
# Polyester is a GuardedPool rather than a CountedPool: there is no global count to snapshot
# and restore, and hence none of the interleaving hazard that would come with one.
#
# WHY NOT A PARTIAL LIMIT
#
# PolyesterWeave can reserve *some* slots, leaving the rest for nested loops, so an obvious
# refinement is to reserve only the `capacity() ÷ budget` slots the outer loop is using. That
# was implemented and measured, and it is a bad trade. Timing a nested-`@batch` workload
# (outer `Threads.@threads`, inner `@batch` over 16384 elements):
#
#   threads  outer   unbudgeted   all-off     partial
#   8        2        0.00026 s   0.00180 s   0.00067 s   partial 2.7x better than all-off
#   8        8        0.00142 s   0.00187 s   0.00200 s   about the same
#   48       48       0.782   s   0.00621 s   0.00604 s   about the same
#   48       2        0.931   s   0.00182 s   0.583   s   partial 320x WORSE
#
# The last row is the point. At high thread counts a nested `@batch` is catastrophic however
# few workers it is given — the per-worker chunks get small enough that launch and
# synchronisation dominate — so leaving any workers free reopens the pathology this package
# exists to close. All-or-nothing costs a bounded factor in the under-subscribed regime
# (row 1, where the machine has spare cores the outer loop was never going to use) and avoids
# an unbounded one in the saturated regime. That asymmetry decides it.
#
# The guard also parks the workers it reserves. A restricted scope is usually about to open a
# region on Julia's scheduler, and the workers it holds are exactly the idle ones that would
# otherwise keep spinning on Julia's threads (see `_quiesce` below); once they are reserved,
# the quiesce that the budgeted macros issue inside the scope finds none of them free.
function _guard(f::F, budget::Int) where {F}
    masks = _reserve_all()
    try
        _park!(masks)
        return f()
    finally
        _release!(masks)
    end
end

# Polyester's workers are Julia tasks that spin on a state word for ~2^20 `pause()` iterations
# after a `@batch` region ends before parking themselves, so until then they are still
# occupying Julia's threads and a `Threads.@threads` region opened in that window waits for
# them rather than running on them. Parking them at once removes the wait. See `QuiescePool`
# for the measurements and <https://github.com/JuliaSIMD/Polyester.jl/issues/82>.
#
# Only the workers taken from PolyesterWeave's free mask are parked, never all of them as
# `ThreadingUtilities.sleep_all_tasks` does. That function overwrites every worker's function
# slot unconditionally, so when another task has just launched a `@batch` chunk on a worker,
# the worker runs the park request instead of the chunk, the launching task sees it finish,
# and that part of its output is never written. A reserved worker cannot receive a chunk,
# so parking it is safe; a worker some `@batch` holds is busy and needs no parking.
function _quiesce()
    masks = _reserve_all()
    try
        _park!(masks)
    finally
        _release!(masks)
    end
    return nothing
end

_reserve_all() = last(Polyester.PolyesterWeave.request_threads(Threads.nthreads()))
_release!(masks) = foreach(Polyester.PolyesterWeave.free_threads!, masks)

# Mirrors `ThreadingUtilities.sleep_all_tasks`, restricted to the workers set in `masks`: bit
# `b` of the `j`-th mask is worker `64(j - 1) + b + 1`, the numbering `@batch` launches with.
# Each park request is issued before any is waited for, so the workers park in parallel.
function _park!(masks)
    fptr = @cfunction(Polyester.ThreadingUtilities._sleep, Cvoid, (Ptr{UInt},))
    _foreach_worker(masks) do tid
        p = Polyester.ThreadingUtilities.taskpointer(tid)
        Polyester.ThreadingUtilities.store!(p, fptr, sizeof(UInt))
        Polyester.ThreadingUtilities._atomic_cas_cmp!(
            p, Polyester.ThreadingUtilities.SPIN, Polyester.ThreadingUtilities.TASK
        )
    end
    _foreach_worker(Polyester.ThreadingUtilities.wait, masks)
    return nothing
end

function _foreach_worker(f::F, masks) where {F}
    offset = 0
    for m in masks
        while !iszero(m)
            tz = trailing_zeros(m)
            f(offset + tz + 1)
            m &= m - one(m)
        end
        offset += 8 * sizeof(UInt)
    end
    return nothing
end

function __init__()
    NestedThreading.register_guarded_pool!(_guard; name = :polyester)
    return NestedThreading.register_quiesce_pool!(_quiesce; name = :polyester)
end

end # module NestedThreadingPolyesterExt
