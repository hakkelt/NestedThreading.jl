# Adding a library

A library joins the registry through one call to [`NestedThreading.register_pool!`](@ref),
made from `__init__`. The call names the library and passes every control it has as a
keyword:

| keyword | called as | what it is for |
| --- | --- | --- |
| `get`, `set` | `get()`, `set(n::Int)` | a persistent thread count |
| `guard` | `guard(f, n::Int)` | a scoped switch with no imperative form |
| `quiesce` | `quiesce()` | make idle workers let go of Julia's threads |
| `park` | `park()` | release workers after a grant lowers the count; needs `get`/`set` |

Pass only the controls the library has. Registration is checked strictly: a name that is
already registered, `get` without `set` (or the reverse), `park` without `get`/`set`, no
control at all, or a `get()` that does not return an `Integer` throws an `ArgumentError`.

## Thread counts

Anything with an imperative thread-count setter:

```julia
function __init__()
    NestedThreading.register_pool!(:mylib; get = MyLib.get_threads, set = MyLib.set_threads)
end
```

The count then takes part in the refcounted snapshot/restore (see
[Composition rules](@ref)), so it is safe under concurrency without further work.

**A boolean toggle is a count too.** Map it onto a count rather than reaching for a guard.
The NFFT extension is written this way:

```julia
_full() = max(2, NestedThreading.capacity())
_get_threads() = MyLib.threading_enabled[] ? _full() : 1
_set_threads(n) = (MyLib.threading_enabled[] = n >= _full())

NestedThreading.register_pool!(:mylib; get = _get_threads, set = _set_threads)
```

The threshold is `capacity()` rather than `1` because a library with no partial control
should be off whenever anything is restricted; see [Composition rules](@ref).

!!! warning "Clamp the threshold to 2"
    The getter and setter must round-trip: `set(get())` has to be a no-op, because that is
    exactly what the last scope exit does. With a bare `capacity()` threshold that breaks in
    a single-threaded session, where `capacity() == 1` makes "restricted to 1" and
    "unrestricted" the same number: the getter maps `false` to 1 and the setter maps 1 back
    to `true`, so every scope silently enables a library the user had switched off and
    never restores it. `max(2, capacity())` is identical everywhere else and costs only
    that `with_threads_enabled` will not enable the library on a machine with one thread,
    where its threaded path has no workers anyway.

## Guards

Only for libraries whose sole control is a scoped context manager with no imperative form.
Polyester's `disable_polyester_threads` is the case this exists for:

```julia
# `guard` is only called while a limit below `capacity()` is in force; `n` is that limit,
# passed in case the library has a proportional control.
_guard(f, n::Int) = MyLib.without_threads(f)

NestedThreading.register_pool!(:mylib; guard = _guard)
```

Switching the library fully off is usually right even when a partial limit is available; see
[Composition rules](@ref) for the measurements behind that.

!!! warning "Do not write a guard that saves and restores a global"
    A guard of the form `prev = X[]; X[] = new; try f() finally X[] = prev end` reintroduces
    exactly the interleaving bug described in [Composition rules](@ref): two concurrent
    guards capture each other's temporary value. If the library has an imperative setter,
    register `get`/`set` instead and let the registry do the bookkeeping. Polyester is safe
    because `disable_polyester_threads` *reserves* worker threads rather than saving and
    restoring a count.

## Quiescing

A separate question from "how many threads may this library use": does it let go of Julia's
threads once its own parallel region has ended? Polyester's workers do not. They are Julia
tasks that spin for about 2^20 `pause()` iterations before parking, and a `Threads.@threads`
region opened during that window waits for them instead of running on them. On 8 threads with
an empty loop body that is 5.7 µs against 219.9 µs.

A `quiesce` hook makes the library release them, and this package's loop macros call it
before every parallel region they open, except a library's own: around a `Polyester.@batch`
loop every hook but Polyester's is called. [`NestedThreading.quiesce_foreign_pools`](@ref)
calls them all. A library with both a guard and such
workers passes both in the same call, as the PolyesterWeave extension does:

```julia
NestedThreading.register_pool!(:mylib; guard = _guard, quiesce = MyLib.park_workers!)
```

The hook takes no arguments, has no paired restore, and must be cheap: the Polyester one is
`ThreadingUtilities.sleep_all_tasks()` at about 0.1 µs, and re-waking the workers for the next
`@batch` costs nothing measurable. A library whose workers park themselves promptly needs no
`quiesce` hook.

## Parking

A library whose workers keep spinning after its thread count is lowered can pass a `park`
hook alongside `get`/`set`. It runs when a [`with_thread_grant`](@ref) closes and lowers
the count, under the registry lock. This package registers one for BLAS, which shuts
OpenBLAS's workers down. For a library of your own:

```julia
NestedThreading.register_pool!(
    :mylib; get = MyLib.get_threads, set = MyLib.set_threads, park = MyLib.release_workers
)
```

The rules a `park` hook must follow, and the cases where it is held back, are in the
extended help of [`NestedThreading.register_pool!`](@ref).

## Doing it from an extension

Registration belongs in `__init__` so that it happens exactly once, when both your package
and the library are loaded. If you own neither package, a small extension in your own package
works the same way:

```toml
[weakdeps]
MyLib = "..."

[extensions]
MyPkgMyLibExt = "MyLib"
```

```julia
module MyPkgMyLibExt
import MyLib, NestedThreading
__init__() = NestedThreading.register_pool!(
    :mylib; get = MyLib.get_threads, set = MyLib.set_threads
)
end
```

An extension may load while a scope is open; the new count is then brought under the limit in
force at once and restored with the others.

Each name can be registered once. A second package that registers the same name throws in its
`__init__`, and so fails to load. Register a library this package does not cover from one
place only; [`thread_pool_info`](@ref) lists what is registered already.
