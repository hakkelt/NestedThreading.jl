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

"""
    QuiescePool(name::Symbol, quiesce)

A threaded library whose workers keep *occupying* Julia's threads after one of its parallel
regions has finished, and which can be told to let go. `quiesce()` is called immediately
before this package opens a region on Julia's own scheduler.

Polyester is the case this exists for. Its workers are Julia tasks that spin on a state word
for about 2^20 `pause()` iterations before parking, so for the whole of that window the Julia
threads they sit on are busy. A `Threads.@threads` region opened in that window does not get
those threads; it waits for them. Measured on an AMD EPYC 7352, 8 Julia threads, Julia 1.13.0,
2026-09-21, with an empty loop body:

    Threads.@threads, in a process that has never run a `@batch`     5.7 us
    Threads.@threads, after one `@batch` has run                   219.9 us
    Threads.@threads, after `@batch` + quiesce                      18.6 us
    the quiesce call itself                                          0.1 us
    Polyester.@batch, after a quiesce                                5.0 us

A 38x penalty, removed for a tenth of a microsecond, and re-waking the workers for the next
`@batch` costs nothing measurable. See the Polyester extension and
<https://github.com/JuliaSIMD/Polyester.jl/issues/82>.

Unlike [`GuardedPool`](@ref) this is not scoped and takes no budget: it is a one-shot "release
the threads now" with no paired restore, because the library re-acquires what it needs at its
next region on its own.

Register with [`register_quiesce_pool!`](@ref).
"""
struct QuiescePool
    name::Symbol
    quiesce::Function
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
const QUIESCE_POOLS = QuiescePool[]

# Parallel to COUNTED_POOLS:
const MAXIMA = Int[]   # each pool's count at registration time (its "full throttle" value)
const SAVED = Int[]    # snapshot taken on the empty -> non-empty ACTIVE transition

"""
    ParkHook(name::Symbol, park)

A function that makes the [`CountedPool`](@ref) `name` release the cores its idle workers
are still holding, called after a [`with_thread_grant`](@ref) scope that had raised that
pool closes and lowers it again.

OpenBLAS is the case this exists for. Lowering its thread count does not stop the workers
it no longer uses: they go on spinning for `OPENBLAS_THREAD_TIMEOUT` (by default 2^28
cycles, about 0.1 s) before they sleep, on cores the caller's own threads now want. A
grant that raises BLAS for one factorization inside a solve that otherwise runs it serial
would leave that spin behind after every factorization.

Register with [`register_park_hook!`](@ref).
"""
struct ParkHook
    name::Symbol
    park::Function
end

"""
    ActiveRestriction(id, budget, exclude, only, kind)

One entry of the [`ACTIVE`](@ref) multiset: a currently-open budget scope.

`id` is a strictly-increasing tag identifying this particular `_enter!` call (so `_exit!`
can remove exactly the right entry without comparing `exclude`/`only` tuples against each
other; see `_exit!`), `exclude` is a denylist of pool names this restriction does not apply
to, and `only` is either `nothing` (applies to every pool) or an allowlist of the only names
it applies to.

`kind` says how the entry combines with the others that apply to the same pool:

* `:limit` — a hard limit, from [`with_thread_budget`](@ref). Nothing nested inside it can
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

const PARK_HOOKS = ParkHook[]

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

"""
    register_quiesce_pool!(quiesce; name::Symbol)

Register a library whose workers hold onto Julia's threads between its own parallel regions.
`quiesce` is called with no arguments before this package opens a region on Julia's
scheduler, and must make that library release them; see [`QuiescePool`](@ref) for why this
exists and what it is worth.

```julia
NestedThreading.register_quiesce_pool!(MyLib.park_workers!; name = :mylib)
```

Registering a `name` that is already present is a no-op. Registration is expected to happen at
load time (`__init__`).
"""
function register_quiesce_pool!(quiesce::Function; name::Symbol)
    @lock REGISTRY_LOCK begin
        if !any(p -> p.name === name, QUIESCE_POOLS)
            push!(QUIESCE_POOLS, QuiescePool(name, quiesce))
        end
    end
    return nothing
end

"""
    register_park_hook!(park; name::Symbol)

Register `park()` to be called whenever a [`with_thread_grant`](@ref) scope closes and the
[`CountedPool`](@ref) `name` goes down as a result, so that the library can release the cores
its now-unused workers would otherwise go on spinning on; see [`ParkHook`](@ref).

`park` is called with no arguments, after the pool's count has already been lowered, with
the registry lock held: no grant can open and start using the workers while they are being
released. It must therefore not wait on another task that opens a budget scope. It is not
called when a plain [`with_thread_budget`](@ref) or [`with_thread_default`](@ref) scope closes:
those lower nothing that was not lowered before they opened.

!!! warning "Calls that bypass the scopes"
    The lock only orders the hook against other scopes. A task that calls into the library
    with no scope of its own, at a count a grant raised, may still be inside that call when
    the grant closes and the hook runs. Whether that is safe is the library's business; for
    OpenBLAS, see the hook this package registers for `:blas`.

Registering a hook for a `name` that already has one is a no-op. Registration is expected to
happen at load time (`__init__`).
"""
function register_park_hook!(park::Function; name::Symbol)
    @lock REGISTRY_LOCK begin
        if !any(h -> h.name === name, PARK_HOOKS)
            push!(PARK_HOOKS, ParkHook(name, park))
        end
    end
    return nothing
end

"""
    quiesce_foreign_pools()

Ask every registered [`QuiescePool`](@ref) to release Julia's worker threads, so that a region
about to be opened on Julia's own scheduler can actually use them.

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

The budget pool `name` should run at right now, given the active entries that apply to it
and `default`, its pre-scope, unrestricted value:

1. With a soft default open, the lowest soft default.
2. Otherwise, with a hard limit open, the lowest hard limit (which may exceed `default`:
   that is how [`with_full_threads`](@ref) turns a default-off library on).
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
        -> (id::Int, guarded_targets::Vector{Int})

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
function _enter!(budget::Int, exclude::Tuple, only, kind::Symbol = :limit)
    return @lock REGISTRY_LOCK begin
        isempty(ACTIVE) && _snapshot!()
        id = (_NEXT_ACTIVE_ID[] += 1)
        push!(ACTIVE, ActiveRestriction(id, budget, exclude, only, kind))
        _apply!()
        (id, Int[_effective_budget(pool.name, capacity()) for pool in GUARDED_POOLS])
    end
end

"""
    _exit!(id::Int)

Remove the restriction tagged `id` (as returned by [`_enter!`](@ref)) from the active
multiset. Restores the snapshot when the last scope exits, otherwise re-applies what
remains active.

When the entry was a grant, every [`ParkHook`](@ref) whose pool went down as a result is
called afterwards, still under the lock, so that no other grant can open and start using the
workers while they are being released.
"""
function _exit!(id::Int)
    @lock REGISTRY_LOCK begin
        i = findfirst(e -> e.id == id, ACTIVE)
        was_grant = i !== nothing && ACTIVE[i].kind === :grant
        before = was_grant ? _current_budgets() : Int[]
        i === nothing || deleteat!(ACTIVE, i)
        if isempty(ACTIVE)
            _restore!()
        else
            _apply!()
        end
        was_grant && _run_park_hooks(_lowered_park_hooks(before))
    end
    return nothing
end

# The count every counted pool is applied at right now, `COUNTED_POOLS`-aligned. Only called
# with the lock held and at least one entry active.
_current_budgets() =
    Int[_effective_budget(pool.name, SAVED[i]) for (i, pool) in enumerate(COUNTED_POOLS)]

# The park hooks of the pools whose count is now below `before`.
function _lowered_park_hooks(before::Vector{Int})
    hooks = ParkHook[]
    isempty(PARK_HOOKS) && return hooks
    for (i, pool) in enumerate(COUNTED_POOLS)
        now = isempty(ACTIVE) ? SAVED[i] : _effective_budget(pool.name, SAVED[i])
        if now < before[i]
            for hook in PARK_HOOKS
                hook.name === pool.name && push!(hooks, hook)
            end
        end
    end
    return hooks
end

@noinline function _run_park_hooks(hooks::Vector{ParkHook})
    for hook in hooks
        hook.park()
    end
    return nothing
end
