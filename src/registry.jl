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

`guard` must be callable as `guard(f::Function, budget::Int)` and must run `f()` with that
library limited to `budget` threads, returning `f()`'s value. It is only ever called with
`budget < capacity()`; an unrestricted scope skips guards entirely.

Most such libraries can only be switched fully off, which is the right default: a partial
limit was implemented for Polyester (which supports one) and measured 320x *worse* in the
saturated regime, because at high thread counts a nested parallel loop is catastrophic
however few workers it gets. See the Polyester extension for the numbers. `budget` is passed
anyway so a library with a genuinely proportional control can use it.

Unlike [`CountedPool`](@ref) this field cannot be a `FunctionWrapper`: the guard receives an
arbitrary caller closure, whose type is not known until the call site, so one runtime
dispatch per guarded scope is inherent. It happens once per budget scope, not once per loop
iteration.

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

# Multiset of currently-active restrictions. Each entry is `(id, budget, exclude, only)`:
# `id` is a strictly-increasing tag identifying this particular `_enter!` call (so `_exit!`
# can remove exactly the right entry without comparing the — abstractly-typed once pulled
# out of this `Vector` — `exclude`/`only` tuples against each other; see `_exit!`),
# `exclude` is a denylist of pool names this restriction does not apply to, and `only` is
# either `nothing` (applies to every pool) or an allowlist of the only names it applies to.
# A pool's effective budget is the minimum `budget` over entries that apply to it — see
# `_applies`/`_effective_budget` below.
const ACTIVE = Tuple{Int,Int,Tuple,Union{Nothing,Tuple}}[]
const _NEXT_ACTIVE_ID = Ref(0)   # protected by REGISTRY_LOCK, like ACTIVE itself

const REGISTRY_LOCK = ReentrantLock()

"""
    capacity() -> Int

The number of worker threads a parallel loop is assumed to be able to occupy. Used as the
numerator of the automatic budget arithmetic and as the "unrestricted" reference value.

Defined as `Threads.threadpoolsize()`. This is exact for `Threads.@threads` and an
approximation for Polyester's `@batch`; it is a single function so that assumption lives
in one place.
"""
capacity() = threadpoolsize()

# --- registration ---------------------------------------------------------------------

"""
    register_counted_pool!(get, set; name::Symbol)

Register a library controlled by a thread-count getter/setter pair. Called from
`NestedThreading`'s own package extensions, and available to any downstream package that
wants to add a library this package does not ship an extension for:

```julia
NestedThreading.register_counted_pool!(MyLib.get_threads, MyLib.set_threads; name = :mylib)
```

Registering a `name` that is already present is a no-op, so duplicate extensions across
packages are harmless.

Registration is expected to happen at load time (`__init__`), before any concurrent use.
Registering while a budget scope is already open is nevertheless handled: the newcomer's
current count becomes its restore value and the restriction in force is applied to it
immediately, so it is restored along with everything else when the last scope exits.
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
            isempty(ACTIVE) || set(_effective_budget(name, current))
        end
    end
    return nothing
end

"""
    register_guarded_pool!(guard; name::Symbol)

Register a library that only has a scoped on/off switch. `guard` is called as
`guard(f, budget::Int)` and must run `f()` with that library limited to `budget` threads,
returning `f()`'s value; see [`GuardedPool`](@ref) for the full contract.

Registering a `name` that is already present is a no-op, so duplicate extensions across
packages are harmless. Registration is expected to happen at load time (`__init__`); a
guard registered while a budget scope is already open is not retroactively applied to it.
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
#
# Invariant, maintained by `register_counted_pool!` (which pushes to both) and by
# `_snapshot!` (which resizes): `length(SAVED) == length(MAXIMA) == length(COUNTED_POOLS)`,
# and index `i` refers to the same pool in all three.

"""
    _name_in(name::Symbol, names::Tuple) -> Bool

`name in names`, written as an explicit `===`-loop rather than `Base.in`/`==`. `names` is
only ever known abstractly as `Tuple` once pulled out of `ACTIVE` (its concrete length
varies per active restriction), and Julia's generic tuple `==` for two tuples whose
concrete lengths are not statically known can widen to `Union{Missing, Bool}` — the path
meant for tuples that might themselves hold `missing` elements, which pool names never do.
`===` has no such case, so this is provably `Bool`, which `in`/`==` here would not be.
"""
function _name_in(name::Symbol, names::Tuple)
    for n in names
        n === name && return true
    end
    return false
end

"""
    _applies(name::Symbol, exclude::Tuple, only) -> Bool

Whether an active restriction with these `exclude`/`only` fields applies to the pool
`name`: not denylisted, and either no allowlist or named in it.
"""
_applies(name::Symbol, exclude::Tuple, only) =
    !_name_in(name, exclude) && (only === nothing || _name_in(name, only))

"""
    _effective_budget(name::Symbol, default::Int) -> Int

The budget pool `name` should run at right now: the minimum `budget` over active
restrictions that apply to it, or `default` (its pre-scope, unrestricted value) when none
do. This is what makes `exclude`/`only` mean "leave this pool alone" rather than "clamp it
to whatever the scope asked for": a pool with no applicable restriction is left at
`default`, not forced down to it.
"""
function _effective_budget(name::Symbol, default::Int)
    target = default
    found = false
    for (_, budget, exclude, only) in ACTIVE
        if _applies(name, exclude, only)
            target = found ? min(target, budget) : budget
            found = true
        end
    end
    return target
end

@noinline function _apply!()
    for (i, pool) in enumerate(COUNTED_POOLS)
        pool.set(_effective_budget(pool.name, SAVED[i]))
    end
    return nothing
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
        pool.set(SAVED[i])
    end
    return nothing
end

"""
    _enter!(budget::Int, exclude::Tuple, only) -> (id::Int, guarded_targets::Vector{Int})

Push a requested restriction onto the active multiset, apply it to every counted pool, and
return its `id` (to be handed back to [`_exit!`](@ref)) together with the per-
[`GuardedPool`](@ref) budgets ([`GUARDED_POOLS`](@ref)-aligned) that this restriction,
together with every other currently-active one, works out to. A guarded pool's entry is
`capacity()` when nothing currently active restricts it, which is the caller's cue to skip
guarding it entirely.

On the transition from "no scope active" to "one scope active" the current counted-pool
counts are snapshotted so that the *last* exit can restore them.

This refcounting is what makes concurrent use safe: the snapshot is taken exactly once and
restored exactly once, so no task can ever restore a value that was itself already
restricted by another task.
"""
function _enter!(budget::Int, exclude::Tuple, only)
    return @lock REGISTRY_LOCK begin
        isempty(ACTIVE) && _snapshot!()
        id = (_NEXT_ACTIVE_ID[] += 1)
        push!(ACTIVE, (id, budget, exclude, only))
        _apply!()
        (id, Int[_effective_budget(pool.name, capacity()) for pool in GUARDED_POOLS])
    end
end

"""
    _exit!(id::Int)

Remove the restriction tagged `id` (as returned by [`_enter!`](@ref)) from the active
multiset. Restores the snapshot when the last scope exits, otherwise re-applies what
remains active.
"""
function _exit!(id::Int)
    @lock REGISTRY_LOCK begin
        i = findfirst(e -> e[1] == id, ACTIVE)
        i === nothing || deleteat!(ACTIVE, i)
        if isempty(ACTIVE)
            _restore!()
        else
            _apply!()
        end
    end
    return nothing
end
