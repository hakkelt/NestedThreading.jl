module NestedThreadingStridedExt

import NestedThreading
import Strided

# Strided threads `@strided` broadcasts and `mapreduce` with `Threads.@spawn`, splitting
# any array above `Strided.MINTHREADLENGTH` elements into `Strided.get_num_threads()`
# tasks — a process-global count, not Polyester, so the Polyester guard does not reach it.
# `Strided.set_num_threads` warns when asked for more than `Threads.nthreads()`, so the
# count is clamped first.
_set(n::Integer) = Strided.set_num_threads(clamp(Int(n), 1, Threads.nthreads()))

function __init__()
    return NestedThreading.register_counted_pool!(
        Strided.get_num_threads, _set; name = :strided
    )
end

end # module NestedThreadingStridedExt
