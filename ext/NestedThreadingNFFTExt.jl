module NestedThreadingNFFTExt

import NestedThreading
import NFFT

# `NFFT._use_threads[]` is a boolean toggle with an imperative setter, so it is registered
# as a thread count (`get`/`set`) rather than a guard: that routes it through the refcounted
# snapshot/restore in the registry instead of having each scope save and restore the global
# itself (which is precisely the interleaving bug this package fixes).
#
# `_use_threads` is not public NFFT API; this extension depends on it because NFFT offers no
# public way to switch its threading. What driving it buys, measured on an AMD EPYC (64 cores),
# Julia 1.13.1, NFFT 0.14.3, 2026-11-20, for a 256² image and a 256-spoke radial trajectory
# with 512 samples per spoke (median of 7 paired measurements, forward / adjoint):
#
#   switching it on for a single call (`with_threads_enabled`)    -t 8   2.7x / 3.4x faster
#                                                                 -t 60  3.4x / 2.5x faster
#   switching it off inside a saturated `Threads.@threads` loop    -t 8   1.07x / 1.04x faster
#                                                                 -t 60  no difference
#
# The mapping is deliberately all-or-nothing — NFFT has no partial thread count, so it is
# enabled only when nothing is restricting the process at all.
#
# `_full()` is `capacity()` clamped up to 2 rather than `capacity()` itself, which matters
# only in a single-threaded session. There `capacity() == 1`, so "restricted to 1" and "full
# throttle" are the same number: with a bare `capacity()` the getter maps `false` to 1 and
# the setter maps 1 back to `true`, and every budget scope would silently flip a user's
# `NFFT._use_threads[] = false` to `true` and never restore it. Clamping to 2 keeps
# get/set a faithful round trip at every thread count. The cost is that `with_threads_enabled`
# does not enable NFFT in a single-threaded session — where its threaded path has no
# workers to use anyway.
_full() = max(2, NestedThreading.capacity())
_get_threads() = NFFT._use_threads[] ? _full() : 1
_set_threads(n::Integer) = (NFFT._use_threads[] = n >= _full())

function __init__()
    return NestedThreading.register_pool!(:nfft; get = _get_threads, set = _set_threads)
end

end # module NestedThreadingNFFTExt
