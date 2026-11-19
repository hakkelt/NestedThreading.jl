# Pool registry and the refcounted thread-budget scope stack.

# Internal records of the registered libraries, one vector per capability. A library
# registered with several capabilities appears in several vectors under the same name.
#
# The callables are stored untyped. A registry that downstream packages extend at load time
# cannot be concretely typed: the set of pools is only known at runtime, so calling
# `pool.get`/`pool.set`/`pool.guard` is a dynamic dispatch whatever the field type. The
# alternatives are worse:
#
#   * `FunctionWrapper` fields do not remove the dispatch, they relocate it into
#     FunctionWrappers' own `reinit_wrapper` path -- which lazily mutates the stored function
#     pointer, and so is not safe for a registry that is read from many threads.
#   * A small `Union` of concrete function types would close the registry to extension,
#     which is the whole point of the package.
#
# The dispatch is confined to the `@noinline` helpers below, so it never leaks into callers'
# inlined code, and it costs one dispatch per *budget scope* -- once per `mul!` -- never once
# per loop iteration. Separate concretely-typed vectors (rather than one
# `Vector{Union{...}}`) keep the iteration itself inference-friendly.

# A library with a persistent thread count (`get`/`set`), and optionally a `park` hook.
struct CountedPool
    name::Symbol
    get::Any
    set::Any
    park::Any   # `nothing` when the library registered none
end

# A library whose only control is a scoped switch, `guard(f, n)`.
struct GuardedPool
    name::Symbol
    guard::Any
end

# A library whose idle workers can be told to let go of Julia's threads, `quiesce()`.
struct QuiescePool
    name::Symbol
    quiesce::Any
end

# Every registered name, whatever its controls. Replaced, never mutated, under REGISTRY_LOCK,
# so a reader without the lock gets a consistent vector.
const POOL_NAMES = Ref(Symbol[])
const COUNTED_POOLS = CountedPool[]
const GUARDED_POOLS = GuardedPool[]
const QUIESCE_POOLS = QuiescePool[]

# Parallel to COUNTED_POOLS:
const MAXIMA = Int[]   # each pool's count at registration time (its unrestricted value)
const SAVED = Int[]    # snapshot taken on the empty -> non-empty ACTIVE transition

"""
    ActiveRestriction(id, budget, exclude, only, kind)

One entry of the [`ACTIVE`](@ref) multiset: a currently-open budget scope.

`id` is a strictly-increasing tag identifying this particular `_enter!` call (so `_exit!`
can remove exactly the right entry without comparing `exclude`/`only` tuples against each
other; see `_exit!`), `exclude` is a denylist of pool names this restriction does not apply
to, and `only` is either `nothing` (applies to every pool) or an allowlist of the only names
it applies to.

`kind` says how the entry combines with the others that apply to the same pool:

* `:limit` — a hard limit, from [`with_thread_limit`](@ref). Nothing nested inside it can
  exceed it.
* `:default` — a soft default, from [`with_thread_default`](@ref): the count to run at
  unless a grant says otherwise.
* `:grant` — a grant, from [`with_thread_grant`](@ref): a call that is known to be worth
  threading, overriding every soft default but no hard limit.

See `_effective_budget` below for the arithmetic.
"""
struct ActiveRestriction
    id::Int
    budget::Int
    exclude::Tuple
    only::Union{Nothing,Tuple}
    kind::Symbol
end

# Multiset of currently-active restrictions.
const ACTIVE = ActiveRestriction[]
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
    register_pool!(name::Symbol; get = nothing, set = nothing, guard = nothing,
                   quiesce = nothing, park = nothing)

Register the threaded library `name` together with every control it offers, in one call.
`NestedThreading`'s own package extensions use it, and so can any package that adds a
library this package does not ship an extension for:

```julia
function __init__()
    NestedThreading.register_pool!(:mylib; get = MyLib.get_threads, set = MyLib.set_threads)
end
```

The keywords are the controls a library can have; pass the ones it has:

* `get`, `set`: a persistent thread count, read as `get()` and written as `set(n::Int)`, such
  as `BLAS.get_num_threads`/`BLAS.set_num_threads`. The first scope to open snapshots the
  count, every scope change applies the effective limit to it, and the last scope to close
  restores the snapshot.
* `guard`: a scoped switch, called as `guard(f, n::Int)`, that runs `f()` with the library
  limited to `n` threads and returns `f()`'s value. See "Guards" below.
* `quiesce`: called as `quiesce()` right before this package opens a parallel region on
  Julia's own scheduler, to make the library's idle workers let go of Julia's threads. See
  "Quiescing" below.
* `park`: called as `park()` after a [`with_thread_grant`](@ref) scope closes and lowers
  this library's count, to make the workers it no longer uses release the cores they are
  spinning on. Needs `get`/`set`. See "Parking" below.

Registration is checked strictly. It throws an `ArgumentError` for a `name` that is already
registered, for `get` without `set` or `set` without `get`, for `park` without `get`/`set`,
for no control at all, and when `get()` does not return an `Integer`. A failed call registers
nothing.

Registration is expected to happen at load time (`__init__`), but an extension may be loaded
while some task has a scope open. A newly registered count then takes its current value as
its restore value and gets the restriction in force at once, so the last scope to close
restores it with everything else. A guard registered while a scope is open is not applied to
that scope, only to those opened afterwards.

# Extended help

## Counts first

Prefer `get`/`set` whenever the library has an imperative setter, even a boolean one. A
count goes through this package's refcounted snapshot/restore and is therefore safe under
concurrency, whereas a guard that saves and restores a global itself reintroduces the very
interleaving bug this package exists to fix. A boolean switch maps onto a count; the NFFT
extension shows the pattern. `guard` is for APIs like Polyester's, which are
reservation-based and have no imperative form.

## Guards

A guard is only ever called with `n < capacity()`; an unrestricted scope skips guards
entirely. Most libraries with only a scoped switch can only be switched fully off, which is
the right default: a partial limit was implemented for Polyester (which supports one) and
measured 320x *worse* in the saturated regime, because at high thread counts a nested
parallel loop is catastrophic however few workers it gets. See the PolyesterWeave extension
for the numbers. `n` is passed anyway so a library with a genuinely proportional control can
use it.

A guard must act for the whole process while it is held, not only for the calling task:
Polyester's reserves its workers, which every task then finds taken. A hard limit nested in
one that already covers this library does not call the guard again (see `_limit_in_force`),
so the enclosing scope's guard is all that restricts a nested call on another task.

The guard receives an arbitrary caller closure, whose type is not known until the call
site, so one runtime dispatch per guarded scope is inherent. It happens once per scope, not
once per loop iteration.

## Quiescing

Some libraries' workers keep *occupying* Julia's threads after one of their parallel regions
has finished. Polyester is the case this exists for. Its workers are Julia tasks that spin
on a state word for about 2^20 `pause()` iterations before parking, so for the whole of that
window the Julia threads they sit on are busy. A `Threads.@threads` region opened in that
window does not get those threads; it waits for them. Measured on an AMD EPYC 7352, 8 Julia
threads, Julia 1.13.0, 2026-09-21, with an empty loop body:

    Threads.@threads, in a process that has never run a `@batch`     5.7 us
    Threads.@threads, after one `@batch` has run                   219.9 us
    Threads.@threads, after `@batch` + quiesce                      18.6 us
    the quiesce call itself                                          0.1 us
    Polyester.@batch, after a quiesce                                5.0 us

A 38x penalty, removed for a tenth of a microsecond, and re-waking the workers for the next
`@batch` costs nothing measurable. See the PolyesterWeave extension and
<https://github.com/JuliaSIMD/Polyester.jl/issues/82>.

Unlike a guard, `quiesce` is not scoped and takes no thread count: it is a one-shot "release
the threads now" with no paired restore, because the library re-acquires what it needs at
its next region on its own. The loop macros call every registered `quiesce` themselves,
except around a `Polyester.@batch` loop.

## Parking

OpenBLAS is the case `park` exists for. Lowering its thread count does not stop the workers
it no longer uses: they go on spinning for `OPENBLAS_THREAD_TIMEOUT` (by default 2^28
cycles, about 0.1 s) before they sleep, on cores the caller's own threads now want. A grant
that raises BLAS for one factorization inside a solve that otherwise runs it serial would
leave that spin behind after every factorization.

`park` is called after the count has already been lowered, with the registry lock held: no
grant can open and start using the workers while they are being released. It must therefore
not wait on another task that opens a scope. It is not called when a plain
[`with_thread_limit`](@ref) or [`with_thread_default`](@ref) scope closes: those lower
nothing that was not lowered before they opened.

It is not called either while a hard limit strictly between 1 and [`capacity`](@ref) is
open around the library. Such a limit is what [`@budgeted_threads`](@ref) opens for a loop
with fewer items than threads, whose other workers may be inside a threaded call to the same
library, at the count a grant raised, when this grant closes.

!!! warning "Calls that bypass the scopes"
    The lock only orders `park` against other scopes. A task that calls into the library
    with no scope of its own, at a count a grant raised, may still be inside that call when
    the grant closes and `park` runs. Whether that is safe is the library's business; for
    OpenBLAS it is not, which is why its hook is skipped under a partial limit.
"""
function register_pool!(
    name::Symbol; get = nothing, set = nothing, guard = nothing, quiesce = nothing,
    park = nothing,
)
    _check_controls(name, get, set, guard, quiesce, park)
    scope_open = @lock REGISTRY_LOCK begin
        name in POOL_NAMES[] && throw(
            ArgumentError(
                "NestedThreading.register_pool!: a pool named $(repr(name)) is already " *
                    "registered. Each library is registered once, with all its controls in " *
                    "one call; pick another name for a different library."
            )
        )
        current = 0
        if get !== nothing
            value = get()
            value isa Integer || throw(
                ArgumentError(
                    "NestedThreading.register_pool!: `get()` of pool $(repr(name)) " *
                        "returned $(repr(value)) of type $(typeof(value)); it must return " *
                        "the library's current thread count as an `Integer`."
                )
            )
            current = Int(value)
        end
        POOL_NAMES[] = [POOL_NAMES[]; name]
        if get !== nothing
            push!(COUNTED_POOLS, CountedPool(name, get, set, park))
            push!(MAXIMA, current)
            # Keep SAVED aligned with COUNTED_POOLS even when registering mid-scope, and
            # apply the restriction currently in force to the newcomer.
            push!(SAVED, current)
            isempty(ACTIVE) || set(_effective_budget(name, current))
        end
        guard === nothing || push!(GUARDED_POOLS, GuardedPool(name, guard))
        quiesce === nothing || push!(QUIESCE_POOLS, QuiescePool(name, quiesce))
        !isempty(ACTIVE)
    end
    scope_open && guard !== nothing &&
        @debug "Guard of pool $(repr(name)) registered while a scope is open; it applies only to scopes opened from now on."
    return nothing
end

function _check_controls(name::Symbol, get, set, guard, quiesce, park)
    prefix = "NestedThreading.register_pool!: pool $(repr(name))"
    (get === nothing) == (set === nothing) || throw(
        ArgumentError(
            "$prefix has $(get === nothing ? "`set` but no `get`" : "`get` but no `set`"); " *
                "a thread count needs both, to be snapshotted and restored."
        )
    )
    park === nothing || get !== nothing || throw(
        ArgumentError(
            "$prefix has `park` but no `get`/`set`; `park` runs when a grant lowers " *
                "the library's thread count, so it needs a count to watch."
        )
    )
    all(isnothing, (get, guard, quiesce)) && throw(
        ArgumentError(
            "$prefix has no control; pass at least one of `get`/`set`, `guard` or " *
                "`quiesce`."
        )
    )
    return nothing
end

"""
    quiesce_foreign_pools()

Call every registered `quiesce` hook (see [`register_pool!`](@ref)), so that a region
about to be opened on Julia's own scheduler can actually use Julia's worker threads.

[`@budgeted_threads`](@ref) and [`@budgeted`](@ref) call this for themselves, except when the
loop they are wrapping is a `Polyester.@batch` — there the Polyester pool *is* the loop. Call
it directly before a hand-written `Threads.@threads` or `@spawn` region that does not go
through those macros.

It is a plain call with no paired restore, costs about 0.1 µs with Polyester loaded, and is a
no-op when nothing is registered. It is safe to call when no foreign region has run: parked
workers stay parked.
"""
@noinline function quiesce_foreign_pools()
    for pool in QUIESCE_POOLS
        pool.quiesce()
    end
    return nothing
end

# --- introspection ---------------------------------------------------------------------

const PoolInfo = NamedTuple{
    (
        :name, :has_count, :has_guard, :has_quiesce, :has_park, :current, :unrestricted,
        :effective,
    ),
    Tuple{Symbol,Bool,Bool,Bool,Bool,Union{Int,Nothing},Union{Int,Nothing},Union{Int,Nothing}},
}

"""
    thread_pool_info() -> Vector{<:NamedTuple}
    thread_pool_info(name::Symbol) -> Union{NamedTuple, Nothing}

Describe every registered library, in registration order, as one `NamedTuple` each, or
only the library registered as `name`. Each has these fields:

* `name::Symbol`: the name it was registered under (see [`register_pool!`](@ref)).
* `has_count`, `has_guard`, `has_quiesce`, `has_park`: which controls it registered
  (`get`/`set`, `guard`, `quiesce`, `park`), each a `Bool`.
* `current`: its thread count right now, as its `get()` reports it.
* `unrestricted`: the count it returns to when no scope is open: the snapshot taken when
  the first open scope entered, or `current` when none is open.
* `effective`: the count the open scopes apply to it, or `current` when none is open.

The three counts are `Int`s, or `nothing` for a library without `get`/`set`. The result is a
snapshot taken under the registry lock; it does not change as scopes open and close.

`thread_pool_info(name)` returns `nothing` for a name no library has registered (logged at
debug level), so a caller can ask about a library whose package may not be loaded.

```julia
# The thread count of every library that has one, right now:
[p.name => p.current for p in thread_pool_info() if p.has_count]

# The count the open scopes apply to BLAS, or `nothing` if BLAS has no count:
info = thread_pool_info(:blas)
info === nothing || info.effective
```
"""
thread_pool_info() = @lock REGISTRY_LOCK PoolInfo[_pool_info(name) for name in POOL_NAMES[]]

function thread_pool_info(name::Symbol)
    info = @lock REGISTRY_LOCK name in POOL_NAMES[] ? _pool_info(name) : nothing
    info === nothing &&
        @debug "thread_pool_info: no pool named $(repr(name)) is registered."
    return info
end

# The `thread_pool_info` entry of the registered pool `name`. Called with the lock held.
function _pool_info(name::Symbol)
    i = findfirst(p -> p.name === name, COUNTED_POOLS)
    current = unrestricted = effective = nothing
    has_park = false
    if i !== nothing
        pool = COUNTED_POOLS[i]
        current = Int(pool.get())::Int
        has_park = pool.park !== nothing
        if isempty(ACTIVE)
            unrestricted = effective = current
        else
            unrestricted = SAVED[i]
            effective = _effective_budget(name, SAVED[i])
        end
    end
    return PoolInfo(
        (
            name, i !== nothing, any(p -> p.name === name, GUARDED_POOLS),
            any(p -> p.name === name, QUIESCE_POOLS), has_park, current, unrestricted,
            effective,
        )
    )
end

# --- counted-pool state transitions ---------------------------------------------------
#
# All of these run with REGISTRY_LOCK held.
#
# Invariant, maintained by `register_pool!` (which pushes to all three) and by
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

The budget pool `name` should run at right now, given the active entries that apply to it
and `default`, its pre-scope, unrestricted value:

1. With a soft default open, the lowest soft default.
2. Otherwise, with a hard limit open, the lowest hard limit (which may exceed `default`:
   that is how [`with_threads_enabled`](@ref) turns a default-off library on).
3. Otherwise `default`.

With a grant open, that is raised to the highest grant, but never past `default`: a grant
restores threading a soft default took away, it neither lowers a pool nor raises it past what
the process runs it at when nothing is open. Whichever it is, it is then clamped to the lowest
hard limit, so nothing can widen past a limit that is open around it.

A pool with no applicable entry is left at `default`, not forced down to it, which is what
makes `exclude`/`only` mean "leave this pool alone" rather than "clamp it to whatever the
scope asked for".
"""
function _effective_budget(name::Symbol, default::Int)
    limit = typemax(Int)
    has_limit = false
    grant = 0
    has_grant = false
    soft = typemax(Int)
    has_soft = false
    for restriction in ACTIVE
        _applies(name, restriction.exclude, restriction.only) || continue
        kind = restriction.kind
        if kind === :grant
            grant = max(grant, restriction.budget)
            has_grant = true
        elseif kind === :default
            soft = min(soft, restriction.budget)
            has_soft = true
        else
            limit = min(limit, restriction.budget)
            has_limit = true
        end
    end
    target = has_soft ? soft : has_limit ? limit : default
    has_grant && (target = max(target, min(grant, default)))
    return min(target, limit)
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
    _enter!(budget::Int, exclude::Tuple, only, kind::Symbol = :limit)
        -> (id::Int, guards::Vector{Tuple{GuardedPool,Int}})

Push a requested restriction onto the active multiset, apply it to every counted pool, and
return its `id` (to be handed back to `_exit!`) together with the guarded pools to invoke
and the budget each gets: those whose budget under this restriction and every other
currently-active one is below `capacity()`. The list is copied under the lock, so a guard
registered while the scope is open does not change it.

On the transition from "no scope active" to "one scope active" the current counted-pool
counts are snapshotted so that the *last* exit can restore them.

This refcounting is what makes concurrent use safe: the snapshot is taken exactly once and
restored exactly once, so no task can ever restore a value that was itself already
restricted by another task.
"""
function _enter!(budget::Int, exclude::Tuple, only, kind::Symbol = :limit)
    return @lock REGISTRY_LOCK begin
        isempty(ACTIVE) && _snapshot!()
        id = (_NEXT_ACTIVE_ID[] += 1)
        push!(ACTIVE, ActiveRestriction(id, budget, exclude, only, kind))
        _apply!()
        guards = Tuple{GuardedPool,Int}[]
        for pool in GUARDED_POOLS
            target = _effective_budget(pool.name, capacity())
            target < capacity() && push!(guards, (pool, target))
        end
        (id, guards)
    end
end

"""
    _exit!(id::Int)

Remove the restriction tagged `id` (as returned by [`_enter!`](@ref)) from the active
multiset. Restores the snapshot when the last scope exits, otherwise re-applies what
remains active.

When the entry was a grant, the `park` hook of every counted pool that went down as a result
is called afterwards, still under the lock, so that no other grant can open and start using the
workers while they are being released.
"""
function _exit!(id::Int)
    not_parked = @lock REGISTRY_LOCK begin
        i = findfirst(e -> e.id == id, ACTIVE)
        was_grant = i !== nothing && ACTIVE[i].kind === :grant
        before = was_grant ? _current_budgets() : Int[]
        i === nothing || deleteat!(ACTIVE, i)
        if isempty(ACTIVE)
            _restore!()
        else
            _apply!()
        end
        skipped = Symbol[]
        was_grant && _run_park_hooks(_lowered_park_hooks(before, skipped))
        skipped
    end
    # Logged after the lock is released, so that a logger never runs while it is held.
    for name in not_parked
        @debug "Pool $(repr(name)) not parked after its grant closed: a hard limit between 1 and capacity() is open around it, so another task may be inside a threaded call to it."
    end
    return nothing
end

# The hard limits opened by the scopes that enclose the calling task, innermost last.
# A `ScopedValue`, so a task spawned inside a scope -- each worker of a `@budgeted_threads`
# loop -- inherits its parent's, while a scope another, unrelated task has open is not in it.
# That distinction is the whole point: only an enclosing scope is guaranteed to stay open
# while the calling task runs. `Base.ScopedValues` arrived in Julia 1.11; on 1.10 nothing is
# recorded, which turns the check below off rather than making it guess.
@static if VERSION >= v"1.11"
    const _ENCLOSING_LIMITS = Base.ScopedValues.ScopedValue{Vector{ActiveRestriction}}(
        ActiveRestriction[]
    )
    _enclosing_limits() = _ENCLOSING_LIMITS[]
    _with_enclosing_limit(f::F, r::ActiveRestriction) where {F} =
        Base.ScopedValues.with(f, _ENCLOSING_LIMITS => [_ENCLOSING_LIMITS[]; r])
else
    _enclosing_limits() = ActiveRestriction[]
    _with_enclosing_limit(f::F, ::ActiveRestriction) where {F} = f()
end

"""
    thread_limit(name::Symbol) -> Union{Int,Nothing}

The hard limit the scopes enclosing the calling task put on pool `name`: the smallest `n`
among the [`with_thread_limit`](@ref) scopes (and so [`with_threads_disabled`](@ref), a
limit of 1) that apply to it and that this task runs inside -- opened by it, or by a task it
was spawned from while the scope was open -- or `nothing` when there is none. A scope
open on an unrelated task does not count: it may close at any moment.

Soft defaults ([`with_thread_default`](@ref)) and grants ([`with_thread_grant`](@ref)) are not
counted, since neither bounds the count. The name does not have to be registered.

`with_thread_limit` already uses this to skip a nested limit that could change nothing, so a
caller needs it only to ask about one pool in particular -- to skip a scope where a limit on
that pool alone is in force, say.

On Julia 1.10, which has no `ScopedValue`, enclosing scopes are not tracked and this always
returns `nothing`: a caller that acts on it then enters its scope, as it would anyway.
"""
function thread_limit(name::Symbol)
    name in POOL_NAMES[] ||
        @debug "thread_limit: no pool named $(repr(name)) is registered; reporting the limits that would apply to it."
    limit = _hard_limit(name, _enclosing_limits())
    return limit == typemax(Int) ? nothing : limit
end

function _hard_limit(name::Symbol, limits::Vector{ActiveRestriction})
    limit = typemax(Int)
    for restriction in limits
        _applies(name, restriction.exclude, restriction.only) || continue
        limit = min(limit, restriction.budget)
    end
    return limit
end

"""
    _limit_in_force(budget::Int, exclude::Tuple, only) -> Bool

Whether a hard limit of `budget` with these `exclude`/`only` fields would be a no-op for the
calling task: some enclosing scope holds every registered counted and guarded pool the new
limit applies to at or below `budget`.

Such a limit changes no pool's effective budget -- soft defaults and grants are clamped by
the enclosing hard limits, which the new one could only repeat -- and its guards would only
repeat the ones the enclosing scope already holds, so [`with_thread_limit`](@ref) runs its
body without entering it. That is the nested serial call inside a saturated
[`@budgeted_threads`](@ref) loop: entering would take the registry lock, set every counted
pool on the way in and out, and run every guard.

Only enclosing scopes count (see `_ENCLOSING_LIMITS`), since the skipped call relies on the
scope that made it a no-op staying open until it returns. That holds on the task that opened
the scope and on every task its body waits for; a task spawned inside a scope and never
waited for can outlive it, and then runs unrestricted once the scope exits. It also relies
on a guard acting for the whole process while it is held (see [`register_pool!`](@ref)).
"""
function _limit_in_force(budget::Int, exclude::Tuple, only)
    limits = _enclosing_limits()
    isempty(limits) && return false
    held(pool) = !_applies(pool.name, exclude, only) || _hard_limit(pool.name, limits) <= budget
    return @lock REGISTRY_LOCK all(held, COUNTED_POOLS) && all(held, GUARDED_POOLS)
end

# The count every counted pool is applied at right now, `COUNTED_POOLS`-aligned. Only called
# with the lock held and at least one entry active.
_current_budgets() =
    Int[_effective_budget(pool.name, SAVED[i]) for (i, pool) in enumerate(COUNTED_POOLS)]

# Whether a hard limit strictly between 1 and `capacity()` applies to pool `name`: the mark of
# a parallel loop whose workers each got more than one thread, i.e. of sibling tasks that may be
# inside a threaded call to this very pool right now, at a count some grant raised.
function _partial_limit_open(name::Symbol)
    for restriction in ACTIVE
        restriction.kind === :limit || continue
        1 < restriction.budget < capacity() || continue
        _applies(name, restriction.exclude, restriction.only) && return true
    end
    return false
end

# The park hooks of the pools whose count is now below `before`. A pool is not parked while a
# partial hard limit is open around it: parking OpenBLAS under a threaded call on another task
# is not safe, and inside a loop with spare threads per worker such a call may be in flight.
# The names of the pools skipped for that reason are pushed to `skipped`.
function _lowered_park_hooks(before::Vector{Int}, skipped::Vector{Symbol})
    hooks = Any[]
    for (i, pool) in enumerate(COUNTED_POOLS)
        pool.park === nothing && continue
        now = _effective_budget(pool.name, SAVED[i])
        now < before[i] || continue
        if _partial_limit_open(pool.name)
            push!(skipped, pool.name)
        else
            push!(hooks, pool.park)
        end
    end
    return hooks
end

# Debug-log the names in `exclude`/`only` that no library is registered under. They are
# legal, since the library may register later, but a typo would otherwise go unnoticed.
@noinline function _debug_unknown_names(exclude::Tuple, only)
    registered = POOL_NAMES[]
    for name in (exclude..., something(only, ())...)
        name in registered ||
            @debug "No pool named $(repr(name)) is registered; `exclude`/`only` ignores the name until a library registers under it."
    end
    return nothing
end

@noinline function _run_park_hooks(hooks::Vector{Any})
    for park in hooks
        park()
    end
    return nothing
end
