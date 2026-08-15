# Adding a library

A library is registered as one of two kinds of *pool*.

## Counted pools

Anything with an imperative thread-count setter:

```julia
function __init__()
    NestedThreading.register_counted_pool!(
        MyLib.get_threads, MyLib.set_threads; name = :mylib
    )
end
```

The pool then participates in the refcounted snapshot/restore, so it is safe under
concurrency for free.

**A boolean toggle is a counted pool too.** Map it onto a count rather than reaching for a
guard — this is how the NFFT extension is written:

```julia
_full() = max(2, NestedThreading.capacity())
_get_threads() = MyLib.threading_enabled[] ? _full() : 1
_set_threads(n) = (MyLib.threading_enabled[] = n >= _full())

NestedThreading.register_counted_pool!(_get_threads, _set_threads; name = :mylib)
```

The threshold is `capacity()` rather than `1` because a library with no partial control
should be off whenever anything is restricted; see [Composition rules](@ref).

!!! warning "Clamp the threshold to 2"
    The getter and setter must round-trip: `set(get())` has to be a no-op, because that is
    exactly what the last scope exit does. With a bare `capacity()` threshold that breaks in
    a single-threaded session, where `capacity() == 1` makes "restricted to 1" and "full
    throttle" the same number — the getter maps `false` to 1 and the setter maps 1 back to
    `true`, so every budget scope silently enables a library the user had switched off and
    never restores it. `max(2, capacity())` is identical everywhere else and costs only
    that `with_full_threads` will not enable the library on a machine with one thread,
    where its threaded path has no workers anyway.

## Guarded pools

Only for libraries whose sole control is a scoped context manager with no imperative form —
Polyester's `disable_polyester_threads` is the motivating case:

```julia
# `guard` is only ever called when a restriction is in force; `budget` is the applied
# thread budget, provided in case the library has a proportional control.
_guard(f, budget::Int) = MyLib.without_threads(f)

NestedThreading.register_guarded_pool!(_guard; name = :mylib)
```

Switching the library fully off is usually right even when a partial limit is available; see
[Composition rules](@ref) for the measurements behind that.

!!! warning "Do not write a guard that saves and restores a global"
    A guard of the form `prev = X[]; X[] = new; try f() finally X[] = prev end` reintroduces
    exactly the interleaving bug described in [Composition rules](@ref) — two concurrent
    guards will capture each other's temporary value. If the library has an imperative
    setter, register it as a counted pool instead and let the registry do the bookkeeping.
    Polyester is safe because `disable_polyester_threads` *reserves* worker threads rather
    than saving and restoring a count.

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
__init__() = NestedThreading.register_counted_pool!(
    MyLib.get_threads, MyLib.set_threads; name = :mylib
)
end
```

Registering the same `name` twice is a no-op, so duplicate extensions across packages are
harmless.
