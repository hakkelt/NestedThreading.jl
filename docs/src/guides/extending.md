# Register a library in NestedThreading

Call [`NestedThreading.register_pool!`](@ref) once, from `__init__`, with every control the
library has:

| keyword | called as | use |
| --- | --- | --- |
| `get`, `set` | `get()`, `set(n::Int)` | a thread count |
| `guard` | `guard(f, n::Int)` | a scoped switch with no setter |
| `quiesce` | `quiesce()` | make idle workers release Julia's threads |
| `park` | `park()` | release workers after a grant lowers the count; needs `get`/`set` |

Registration throws an `ArgumentError` for a name that is already registered, `get` without
`set` or the reverse, `park` without `get`/`set`, no control at all, or a `get()` that does
not return an `Integer`. Each name can be registered from one place only;
[`thread_pool_info`](@ref) lists the names already taken.

## A thread count

```julia
function __init__()
    NestedThreading.register_pool!(:mylib; get = MyLib.get_threads, set = MyLib.set_threads)
end
```

## An on/off switch

Map a boolean onto a count. The NFFT extension does this:

```julia
_full() = max(2, NestedThreading.capacity())
_get_threads() = MyLib.threading_enabled[] ? _full() : 1
_set_threads(n) = (MyLib.threading_enabled[] = n >= _full())

NestedThreading.register_pool!(:mylib; get = _get_threads, set = _set_threads)
```

The library is then off whenever any limit is below `capacity()`.

!!! warning "Keep the threshold at least 2"
    `set(get())` must change nothing, because the last scope to close restores the counts
    that way. With a threshold of `capacity()` alone, a single-threaded session maps "off" to
    1 and 1 back to "on", so every scope turns the library on and leaves it on.

## A scoped switch (guard)

Use `guard` only when the library has no setter, only a function that runs `f` with its
threads off, such as Polyester's `disable_polyester_threads`. A guard is called only while a
limit below `capacity()` is in force, with that limit as `n`:

```julia
_guard(f, n::Int) = MyLib.without_threads(f)

NestedThreading.register_pool!(:mylib; guard = _guard)
```

!!! warning "A guard must not save and restore a global"
    A guard of the form `prev = X[]; X[] = new; try f() finally X[] = prev end` breaks when
    two tasks run it concurrently: each restores the other's temporary value. If the library
    has a setter, register `get`/`set` and let the registry restore the count.

## Workers that keep spinning

If the library's workers keep spinning after a parallel region ends, pass a `quiesce` hook
that makes them release Julia's threads. It takes no arguments, must be cheap, and must be
safe while another task is using the library: it must not take work away from a worker that
another task has just given some. `ThreadingUtilities.sleep_all_tasks()` is an example of an
unsafe hook: a worker that has just been handed another task's `@batch` chunk parks instead,
and that chunk is never computed. The PolyesterWeave extension passes both hooks:

```julia
NestedThreading.register_pool!(:mylib; guard = _guard, quiesce = MyLib.park_workers!)
```

If the workers keep spinning after their count is lowered, pass a `park` hook with
`get`/`set`. It runs when a grant closes and lowers the count; its rules are in the extended
help of [`NestedThreading.register_pool!`](@ref).

## From a package extension

If you own neither package, register from an extension of your own:

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

An extension that loads while a scope is open has its count lowered to the limit in force at
once, and restored with the others.
