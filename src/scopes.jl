# Scoped budget API built on the refcounted registry.

# Compose the guards `_enter!` selected around `f`. Recursive rather than a loop so the
# abstract `guard` call is the only dynamic dispatch and it stays out of the caller's
# inlining. `guards` already reflects every currently-active restriction (including this one)
# and its `exclude`/`only`, and holds only the pools whose budget is below `capacity()`.
#
# A guard that fails is skipped, never allowed to change what the body did: `GuardedCall`
# tracks whether the body was entered and how it ended, so a throw from the guard itself can
# be told apart from one coming out of the body.
mutable struct GuardedCall
    entered::Bool
    returned::Bool
    body_threw::Bool
    value::Any
    exception::Any
    GuardedCall() = new(false, false, false, nothing, nothing)
end

@noinline function _run_guarded(f::F, guards::Vector{Tuple{GuardedPool,Int}}, i::Int) where {F}
    i > length(guards) && return f()
    pool, target = guards[i]
    call = GuardedCall()
    try
        value = pool.guard(target) do
            call.entered = true
            try
                inner = _run_guarded(f, guards, i + 1)
                call.value = inner
                call.returned = true
                return inner
            catch exception
                call.body_threw = true
                call.exception = exception
                rethrow()
            end
        end
        call.entered && !call.body_threw && return value
        _report_hook_failure(
            pool.name, :guard,
            ErrorException(
                call.entered ? "the guard swallowed an exception thrown by the function it guards" :
                    "the guard returned without calling the function it guards"
            ),
            backtrace(),
        )
        call.body_threw && throw(call.exception)
    catch exception
        if call.body_threw
            exception === call.exception && rethrow()
            _report_hook_failure(pool.name, :guard, exception, catch_backtrace())
            throw(call.exception)
        end
        _report_hook_failure(pool.name, :guard, exception, catch_backtrace())
        call.returned && return call.value
    end
    # The guard failed before running the body: run it without this guard.
    return _run_guarded(f, guards, i + 1)
end

"""
    with_thread_limit(f, n::Integer; exclude = (), only = nothing)

Run `f()` with every registered threaded library limited to `n` threads, restoring the
previous state afterwards (also on exception). Returns `f()`'s value.

The limit actually applied to a pool is the `minimum` over every *currently-active*
restriction that applies to that pool (see `exclude`/`only` below), so

* nesting can only ever narrow, never widen, an outer restriction — an inner
  `with_thread_limit(f, 8)` inside an outer `with_thread_limit(g, 1)` still runs at 1;
* concurrent scopes cannot corrupt each other's bookkeeping: the counts are snapshotted on
  the first entry and restored on the last exit.
* a limit nested in an enclosing one -- on the same task, or on a task spawned inside it,
  such as a [`@budgeted_threads`](@ref) worker -- that already holds every pool it covers at
  or below `n` is not entered: it could change nothing, so `f()` runs directly, without
  entering and exiting a scope, setting every pool and running the guards, which is what
  makes a scope cost microseconds. Checking that still takes the registry lock once. A scope open on an
  unrelated task never counts. This relies on the enclosing scope outliving `f()`, which a
  task spawned inside a scope and never waited for does not guarantee. Julia 1.11 and later
  only; on 1.10 every scope is entered.

By default a restriction applies to every registered pool, thread counts and guards alike
(see [`register_pool!`](@ref)). Two keywords narrow that:

* `exclude`: a denylist of pool names this restriction does not apply to. Naming a pool
  leaves its thread count and its guard alone. Naming a pool that is not currently registered at all is not an error: a
  caller may exclude a pool whose package is not loaded in this session.
* `only`: an allowlist — when given, this restriction applies to *only* these pools and
  none else, leaving everything unnamed alone entirely (not merely unrestricted-by-this-
  scope: if no other active scope restricts it either, it keeps its pre-scope value even
  past this scope's own limit). This is what a call site that wants, say, serial BLAS
  needs: "restrict exactly these pools" is not expressible as a denylist without enumerating every
  other pool, which breaks whenever a new one registers.

Each takes a single pool name or a collection of them: `exclude = :polyester`,
`only = (:blas, :mkl)` and `only = [:blas, :mkl]` all work. An element that is not a
`Symbol` throws an `ArgumentError`.

Guards are invoked only when the limit applying to them is below
[`capacity`](@ref), and receive that limit. Polyester's guard switches it off outright
rather than limiting it proportionally — a partial limit was tried and measured far worse
in the saturated regime; see [`register_pool!`](@ref). [`@budgeted_batch`](@ref) uses
`exclude = (:polyester,)` because there the outer loop *is* the Polyester consumer.

!!! note "Guards are decided once, at entry"
    Counted pools follow the applied minimum for as long as the scope is open — another
    task narrowing the limit lowers them immediately. Guards cannot work that way: a
    guard is a lexical wrapper around `f()`, so whether it runs is decided from the limit
    *at entry* and cannot be revised afterwards.

    So a scope that enters while nothing is restricted runs its body unguarded even if a
    concurrent task restricts the process meanwhile: its BLAS and FFTW counts drop, but a
    nested Polyester loop inside it can still spawn workers. The other direction is safe —
    a scope that entered restricted stays guarded for its whole body even after the other
    scope exits. Only the guarded libraries are affected, and only for scopes that opened
    unrestricted; if it matters, open the restricting scope before spawning the tasks
    rather than from inside them.

!!! note "A limit of `capacity()` raises, it does not merely permit"
    The limit is applied to every counted pool this restriction applies to
    unconditionally, so a scope whose budget works out to [`capacity`](@ref) —
    `with_threads_enabled`, or a loop with a single item — sets those pools *up* to that
    value for its duration, even past a lower count the caller had configured
    (`BLAS.set_num_threads(2)`, say). The original counts are restored on exit like any
    other scope. This is what makes `with_threads_enabled` able to turn NFFT on; if a
    hand-tuned lower count must be preserved, do not open an unrestricted scope around it
    — or `exclude`/`only` it out of one.

    A pool named in `exclude`, or left out of `only`, is not "permitted" either way: as
    long as nothing else currently active restricts it, it keeps whatever value it already
    had, not the value this scope requested.

!!! note "Process-global state"
    BLAS and FFTW thread counts are process-global with no per-task scoping. Taking the
    minimum over all active scopes is deliberately conservative: while any task is
    restricted, unrelated concurrent tasks are restricted too. That never oversubscribes,
    but it does mean a long-lived restricted scope throttles the whole process. This is a
    limitation of the underlying C libraries, not something this package can solve.

See also [`with_threads_disabled`](@ref), [`with_threads_enabled`](@ref).
"""
function with_thread_limit(
    f::F, n::Integer; exclude = (), only = nothing
) where {F}
    return _with_scope(f, n, exclude, only, :limit)
end

function _with_scope(f::F, n::Integer, exclude, only, kind::Symbol) where {F}
    exclude = _pool_names(exclude)
    only = only === nothing ? nothing : _pool_names(only)
    budget = max(1, Int(n))
    n < 1 && @debug "Thread count $n raised to 1."
    (isempty(exclude) && only === nothing) || _debug_unknown_names(exclude, only)
    # `guards` reflects every currently-active restriction, computed once at entry under
    # the registry lock. Guards
    # are lexical wrappers and so are decided here, once, and never revised — see the
    # docstring for what that costs. Counted pools go on tracking the live minimum through
    # `_enter!`/`_exit!` for as long as the scope is open.
    # A limit an enclosing scope already enforces could change nothing; see `_limit_in_force`.
    # Not logged: the skip honors the request in full, and a disabled `@debug` would still
    # cost several times this path's own run time.
    kind === :limit && _limit_in_force(budget, exclude, only) && return f()
    id, guards = _enter!(budget, exclude, only, kind)
    return try
        if kind === :limit
            _with_enclosing_limit(ActiveRestriction(id, budget, exclude, only, kind)) do
                _run_guarded(f, guards, 1)
            end
        else
            _run_guarded(f, guards, 1)
        end
    finally
        _exit!(id)
    end
end

# `exclude`/`only` as the `Tuple` of names the registry stores. A tuple of symbols passes
# through unchanged, so the common call allocates nothing here.
_pool_names(names::Tuple{Vararg{Symbol}}) = names
_pool_names(name::Symbol) = (name,)
function _pool_names(names)
    for name in names
        name isa Symbol || throw(
            ArgumentError(
                "pool names in `exclude`/`only` must be `Symbol`s, got $(repr(name))"
            )
        )
    end
    return Tuple(names)::Tuple{Vararg{Symbol}}
end

"""
    with_thread_default(f, n::Integer; exclude = (), only = nothing)

Run `f()` with every registered threaded library at `n` threads *by default*: the count it
runs at unless a [`with_thread_grant`](@ref) nested inside says a particular call is worth
more. Returns `f()`'s value.

This is the scope to open around a whole computation whose calls into a library are mostly
too small to benefit from its threads, but some of which are not. A hard
[`with_thread_limit`](@ref) cannot express that: nothing nested inside it can widen past it,
so the few large calls would run serial too.

Soft defaults combine with each other by minimum, and yield to a grant. They do not yield to
nothing else: a hard limit open around or inside a soft default still clamps it, and a soft
default of `n` below a hard limit of `m > n` runs at `n`, since the limit is a ceiling and not
a request.

```julia
with_thread_default(1; only = (:blas,)) do     # small BLAS calls: serial
    for iteration in 1:100
        axpy!(a, x, y)                          # BLAS at 1 thread
        with_thread_grant(NestedThreading.capacity(); only = (:blas,)) do
            svd!(big)                           # BLAS at every thread
        end
    end
end
```

See [`with_thread_limit`](@ref) for `exclude`, `only` and what is process-global.
"""
function with_thread_default(
    f::F, n::Integer; exclude = (), only = nothing
) where {F}
    return _with_scope(f, n, exclude, only, :default)
end

"""
    with_thread_grant(f, n::Integer; exclude = (), only = nothing)

Run `f()` with every registered threaded library allowed `n` threads *despite* any soft
default in force ([`with_thread_default`](@ref)), for a call that is known to be large
enough to be worth threading. Returns `f()`'s value.

A grant overrides soft defaults but never a hard limit: inside a
[`with_thread_limit`](@ref)`(g, 1)` — a worker of a saturated outer parallel loop, say — a
grant does nothing. It also never raises a pool past the count the process runs it at when
no scope is open: it restores threading that a soft default took away, it does not add
threads a caller turned off by hand with, e.g., `BLAS.set_num_threads(1)`. Several grants open
at once combine by maximum.

When a grant closes and a pool's count goes down as a result, the pool's `park` hook runs
(see [`register_pool!`](@ref)), so that a library whose idle workers would otherwise go on spinning
(OpenBLAS) releases them. That costs something (about a millisecond for OpenBLAS, counting
the pool's restart at the next grant), so a grant belongs around a call that is worth
several milliseconds, not around a small one. Decide whether a call is large enough
*before* opening the scope rather than opening one per call and letting it do nothing: opening a scope costs a lock and a pass over
the registry either way.

See [`with_thread_limit`](@ref) for `exclude`, `only` and what is process-global.
"""
function with_thread_grant(
    f::F, n::Integer; exclude = (), only = nothing
) where {F}
    return _with_scope(f, n, exclude, only, :grant)
end

"""
    with_threads_disabled(f; exclude = (), only = nothing)

Run `f()` with every registered threaded library limited to a single thread. Equivalent to
`with_thread_limit(f, 1)`. Use at call sites that are not a plain `for` loop — a
`@sync`/`@spawn` block, or a single call gated on a `Bool` field.
"""
with_threads_disabled(f::F; exclude = (), only = nothing) where {F} =
    with_thread_limit(f, 1; exclude, only)

"""
    with_threads_enabled(f; exclude = (), only = nothing)

Run `f()` with every registered thread count at [`capacity`](@ref). The scope adds no guard
of its own; a guard runs only if another open restriction lowers the limit below
`capacity()` at entry. Unlike a plain unwrapped call this actively raises the counts, so it turns on a library that
defaults to off (it sets `NFFT._use_threads[] = true`, for instance), which is what a call
site with an explicit `threaded = true` switch needs. Equivalent to
`with_thread_limit(f, capacity())`.

Still clamped by any outer restriction, so calling it inside a saturated batch loop is
safe and does nothing.
"""
with_threads_enabled(f::F; exclude = (), only = nothing) where {F} =
    with_thread_limit(f, capacity(); exclude, only)

"""
    reset_thread_counts!()

Set every registered thread count back to the value it had when its library registered,
both now (if no scope is open) and as the value the last scope to close restores. This is
the process-wide escape hatch for code that only ever runs one thing at a time and wants
every library back at the count it started with, typically unrestricted, after something
lowered it by hand (`BLAS.set_num_threads(1)`, say).

Scopes still win while they are open; this only changes what the process returns to when
nothing is open. It is not scoped and has no inverse.
"""
function reset_thread_counts!()
    failures = @lock REGISTRY_LOCK begin
        for i in eachindex(COUNTED_POOLS)
            SAVED[i] = MAXIMA[i]
        end
        isempty(ACTIVE) && _restore!()
        _take_hook_failures!()
    end
    _report_hook_failures(failures)
    return nothing
end

"""
    budget_for(range) -> Int

The per-worker inner thread budget for a parallel loop over `range`:
`max(1, capacity() ÷ length(range))`.

With `capacity() == 8`, a loop of 32 items gets budget 1 (the outer loop already
saturates the machine) while a loop of 2 items gets budget 4 (2 workers × 4 inner threads).
Iterators of unknown length fall back to the conservative budget of 1.
"""
budget_for(range) = _budget_for(Base.IteratorSize(range), range)
_budget_for(::Union{Base.HasLength,Base.HasShape}, range) =
    max(1, capacity() ÷ max(1, length(range)))
_budget_for(::Base.IteratorSize, _) = 1
