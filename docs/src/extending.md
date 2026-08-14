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
_get_threads() = MyLib.threading_enabled[] ? NestedThreading.capacity() : 1
_set_threads(n) = (MyLib.threading_enabled[] = n >= NestedThreading.capacity())

NestedThreading.register_counted_pool!(_get_threads, _set_threads; name = :mylib)
```

`n >= capacity()` rather than `n > 1` because a library with no partial control should be off
whenever anything is restricted; see [Composition rules](@ref).

## Guarded pools

Only for libraries whose sole control is a scoped context manager with no imperative form —
Polyester's `disable_polyester_threads` is the motivating case:

```julia
_guard(f, restricted::Bool) = restricted ? MyLib.without_threads(f) : f()

NestedThreading.register_guarded_pool!(_guard; name = :mylib)
```

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
