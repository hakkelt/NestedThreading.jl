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
#
# `_full()` is `capacity()` clamped up to 2 rather than `capacity()` itself, which matters
# only in a single-threaded session. There `capacity() == 1`, so "restricted to 1" and "full
# throttle" are the same number: with a bare `capacity()` the getter maps `false` to 1 and
# the setter maps 1 back to `true`, and every budget scope would silently flip a user's
# `NFFT._use_threads[] = false` to `true` and never restore it. Clamping to 2 keeps
# get/set a faithful round trip at every thread count. The cost is that `with_full_threads`
# does not enable NFFT in a single-threaded session — where its threaded path has no
# workers to use anyway.
_full() = max(2, NestedThreading.capacity())
_get_threads() = NFFT._use_threads[] ? _full() : 1
_set_threads(n::Integer) = (NFFT._use_threads[] = n >= _full())

function __init__()
    return NestedThreading.register_counted_pool!(
        _get_threads, _set_threads; name = :nfft
    )
end

end # module NestedThreadingNFFTExt
