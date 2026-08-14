module NestedThreadingNFFTExt

import NestedThreading
import NFFT

# `NFFT._use_threads[]` is a boolean toggle with an imperative setter, so it is registered
# as a CountedPool rather than a GuardedPool: that routes it through the refcounted
# snapshot/restore in the registry instead of having each scope save and restore the global
# itself (which is precisely the interleaving bug this package fixes).
#
# The mapping is deliberately all-or-nothing — NFFT has no partial thread count, so it is
# enabled only when nothing is restricting the process at all.
_get_threads() = NFFT._use_threads[] ? NestedThreading.capacity() : 1
_set_threads(n::Integer) = (NFFT._use_threads[] = n >= NestedThreading.capacity())

function __init__()
    return NestedThreading.register_counted_pool!(
        _get_threads, _set_threads; name = :nfft
    )
end

end # module NestedThreadingNFFTExt
