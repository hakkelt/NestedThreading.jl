# Pool registry and the refcounted thread-budget scope stack.

"""
    CountedPool(name::Symbol, get, set)

A threaded library that exposes a persistent *thread count* through a getter/setter pair,
e.g. `BLAS.get_num_threads`/`BLAS.set_num_threads` or `FFTW.get_num_threads`/
`FFTW.set_num_threads`.

Register with [`register_counted_pool!`](@ref).
"""
struct CountedPool
    name::Symbol
    get::Function
    set::Function
end

"""
    GuardedPool(name::Symbol, guard)

A threaded library whose only control is a *scoped* context manager, e.g. Polyester's
`disable_polyester_threads`.

`guard` must be callable as `guard(f::Function, restricted::Bool)` and must run `f()` with
that library's threading disabled when `restricted` is `true`, returning `f()`'s value.

Prefer [`CountedPool`](@ref) whenever the library exposes an imperative setter, even a
boolean one — a counted pool goes through this package's refcounted snapshot/restore and is
therefore safe under concurrency, whereas a guard that saves and restores a global itself
reintroduces the very interleaving bug this package exists to fix. A boolean toggle maps
cleanly onto a count; see the NFFT extension for the pattern. `GuardedPool` is for APIs like
Polyester's, which are reservation-based and have no imperative form.

Register with [`register_guarded_pool!`](@ref).
"""
struct GuardedPool
    name::Symbol
    guard::Function
end

# Separate concretely-typed vectors (rather than one Vector{Union{...}}) so that iteration
# stays inference-friendly.
#
# ON THE `::Function` FIELDS AND THEIR RUNTIME DISPATCH
#
# A registry that downstream packages extend at load time cannot be concretely typed: the
# set of pools is only known at runtime, so calling `pool.get`/`pool.set`/`pool.guard` is a
# dynamic dispatch. This is inherent, not an oversight, and the alternatives are worse:
#
#   * `FunctionWrapper` fields do not remove the dispatch, they relocate it into
#     FunctionWrappers' own `reinit_wrapper` path — which lazily mutates the stored function
#     pointer, and so is not safe for a registry that is read from many threads.
#   * A small `Union` of concrete function types would close the registry to extension,
#     which is the whole point of the package.
#
# What is done instead: the dispatch is confined to the `@noinline` helpers below, so it
# never leaks into callers' inlined code, and it costs one dispatch per *budget scope* —
# once per `mul!` — never once per loop iteration.
const COUNTED_POOLS = CountedPool[]
const GUARDED_POOLS = GuardedPool[]

# Parallel to COUNTED_POOLS:
const MAXIMA = Int[]   # each pool's count at registration time (its "full throttle" value)
const SAVED = Int[]    # snapshot taken on the empty -> non-empty ACTIVE transition
const ACTIVE = Int[]   # multiset of currently-active requested budgets

const REGISTRY_LOCK = ReentrantLock()

"""
    capacity() -> Int

The number of worker threads a parallel loop is assumed to be able to occupy. Used as the
numerator of the automatic budget arithmetic and as the "unrestricted" reference value.

Defined as `Threads.nthreads()`. This is exact for `Threads.@threads` and an approximation
for Polyester's `@batch`; it is a single function so that assumption lives in one place.
"""
capacity() = nthreads()

_is_active() = @lock REGISTRY_LOCK !isempty(ACTIVE)

# --- registration ---------------------------------------------------------------------

"""
    register_counted_pool!(get, set; name::Symbol)

Register a library controlled by a thread-count getter/setter pair. Called from
`NestedThreading`'s own package extensions, and available to any downstream package that
wants to add a library this package does not ship an extension for:

```julia
NestedThreading.register_counted_pool!(MyLib.get_threads, MyLib.set_threads; name = :mylib)
```

Registration is expected to happen at load time (`__init__`), before any concurrent use.
"""
function register_counted_pool!(get::Function, set::Function; name::Symbol)
    @lock REGISTRY_LOCK begin
        if !any(p -> p.name === name, COUNTED_POOLS)
            push!(COUNTED_POOLS, CountedPool(name, get, set))
            current = Int(get())
            push!(MAXIMA, current)
            # Keep SAVED aligned with COUNTED_POOLS even when registering mid-scope, and
            # apply the restriction currently in force to the newcomer.
            push!(SAVED, current)
            isempty(ACTIVE) || set(minimum(ACTIVE))
        end
    end
    return nothing
end

"""
    register_guarded_pool!(guard; name::Symbol)

Register a library that only has a scoped on/off switch. `guard` is called as
`guard(f, restricted::Bool)`; see [`GuardedPool`](@ref) for the contract.
"""
function register_guarded_pool!(guard::Function; name::Symbol)
    @lock REGISTRY_LOCK begin
        if !any(p -> p.name === name, GUARDED_POOLS)
            push!(GUARDED_POOLS, GuardedPool(name, guard))
        end
    end
    return nothing
end

# --- counted-pool state transitions ---------------------------------------------------
#
# All of these run with REGISTRY_LOCK held.

@noinline function _apply!(n::Int)
    for pool in COUNTED_POOLS
        pool.set(n)
    end
    return n
end

@noinline function _snapshot!()
    resize!(SAVED, length(COUNTED_POOLS))
    for (i, pool) in enumerate(COUNTED_POOLS)
        SAVED[i] = Int(pool.get())::Int
    end
    return nothing
end

@noinline function _restore!()
    for (i, pool) in enumerate(COUNTED_POOLS)
        i <= length(SAVED) && pool.set(SAVED[i])
    end
    return nothing
end

"""
    _enter!(n::Int) -> Int

Push a requested budget onto the active multiset and return the budget actually applied,
which is `minimum(ACTIVE)`. On the transition from "no scope active" to "one scope active"
the current counts are snapshotted so that the *last* exit can restore them.

This refcounting is what makes concurrent use safe: the snapshot is taken exactly once and
restored exactly once, so no task can ever restore a value that was itself already
restricted by another task.
"""
function _enter!(n::Int)
    return @lock REGISTRY_LOCK begin
        isempty(ACTIVE) && _snapshot!()
        push!(ACTIVE, n)
        _apply!(minimum(ACTIVE))
    end
end

"""
    _exit!(n::Int)

Remove one occurrence of `n` from the active multiset. Restores the snapshot when the last
scope exits, otherwise re-applies the new minimum.
"""
function _exit!(n::Int)
    @lock REGISTRY_LOCK begin
        i = findfirst(==(n), ACTIVE)
        i === nothing || deleteat!(ACTIVE, i)
        if isempty(ACTIVE)
            _restore!()
        else
            _apply!(minimum(ACTIVE))
        end
    end
    return nothing
end
