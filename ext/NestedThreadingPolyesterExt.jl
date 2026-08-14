module NestedThreadingPolyesterExt

import NestedThreading
import Polyester

# Polyester has no thread-count setter, so this guard switches it off for the duration of any
# restricted scope. `disable_polyester_threads` does that by reserving every PolyesterWeave
# worker slot, so nested `@batch` loops find none free and run serially. Reservation is
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
function _guard(f::F, budget::Int) where {F}
    return Polyester.disable_polyester_threads(f)
end

function __init__()
    return NestedThreading.register_guarded_pool!(_guard; name = :polyester)
end

end # module NestedThreadingPolyesterExt
