# Scoped budget API built on the refcounted registry.

# Compose every registered guard around `f`. Recursive rather than a loop so the abstract
# `guard` call is the only dynamic dispatch and it stays out of the caller's inlining.
@noinline function _run_guarded(f::F, restricted::Bool, exclude::Tuple, i::Int) where {F}
    i > length(GUARDED_POOLS) && return f()
    pool = GUARDED_POOLS[i]
    if pool.name in exclude
        return _run_guarded(f, restricted, exclude, i + 1)
    end
    return pool.guard(restricted) do
        _run_guarded(f, restricted, exclude, i + 1)
    end
end

"""
    with_thread_budget(f, n::Integer; exclude = ())

Run `f()` with every registered threaded library limited to `n` threads, restoring the
previous state afterwards (also on exception). Returns `f()`'s value.

The budget actually applied is `minimum` over *all* budget scopes active in the process, so

* nesting can only ever narrow, never widen, an outer restriction — an inner
  `with_thread_budget(f, 8)` inside an outer `with_thread_budget(g, 1)` still runs at 1;
* concurrent scopes cannot corrupt each other's bookkeeping: the counts are snapshotted on
  the first entry and restored on the last exit.

[`GuardedPool`](@ref)s (Polyester, NFFT) have no partial-count API, so they are disabled
whenever `n < capacity()` — a budget of 4 inside an 8-thread outer loop must not leave a
nested Polyester loop free to spawn its own workers. Pass pool names in `exclude` to skip
guarding a specific pool; [`@budgeted_batch`](@ref) uses `exclude = (:polyester,)` because
there the outer loop *is* the Polyester consumer.

!!! note "Process-global state"
    BLAS and FFTW thread counts are process-global with no per-task scoping. Taking the
    minimum over all active scopes is deliberately conservative: while any task is
    restricted, unrelated concurrent tasks are restricted too. That never oversubscribes,
    but it does mean a long-lived restricted scope throttles the whole process. This is a
    limitation of the underlying C libraries, not something this package can solve.

See also [`with_restricted_threads`](@ref), [`with_full_threads`](@ref).
"""
function with_thread_budget(f::F, n::Integer; exclude::Tuple = ()) where {F}
    budget = max(1, Int(n))
    applied = _enter!(budget)
    return try
        _run_guarded(f, applied < capacity(), exclude, 1)
    finally
        _exit!(budget)
    end
end

"""
    with_restricted_threads(f; exclude = ())

Run `f()` with every registered threaded library limited to a single thread. Equivalent to
`with_thread_budget(f, 1)`. Use at call sites that are not a plain `for` loop — a
`@sync`/`@spawn` block, or a single call gated on a `Bool` field.
"""
with_restricted_threads(f::F; exclude::Tuple = ()) where {F} =
    with_thread_budget(f, 1; exclude)

"""
    with_full_threads(f; exclude = ())

Run `f()` requesting full threading: every [`CountedPool`](@ref) at [`capacity`](@ref) and
every [`GuardedPool`](@ref) *enabled*. Unlike a plain unwrapped call this actively turns
guarded libraries on (e.g. sets `NFFT._use_threads[] = true`), which is what a call site
with an explicit `threaded = true` switch needs.

Still clamped by any outer restriction, so calling it inside a saturated batch loop is
safe and does nothing.
"""
with_full_threads(f::F; exclude::Tuple = ()) where {F} =
    with_thread_budget(f, capacity(); exclude)

"""
    enable_full_threading()

Permanently set every registered [`CountedPool`](@ref) to its registration-time maximum —
the process-wide escape hatch for code that only ever runs one thing at a time and wants
every library at full throttle.

This sets the *baseline*: the value restored when the last budget scope exits. Scoped
restrictions still win while they are active; this only changes what the process returns
to when nothing is restricted.

Not scoped and not undoable — there is deliberately no `disable_full_threading`.
"""
function enable_full_threading()
    @lock REGISTRY_LOCK begin
        for i in eachindex(COUNTED_POOLS)
            SAVED[i] = MAXIMA[i]
        end
        isempty(ACTIVE) && _restore!()
    end
    return nothing
end

"""
    budget_for(range) -> Int

The per-worker inner thread budget for a parallel loop over `range`:
`max(1, capacity() ÷ length(range))`.

With `Threads.nthreads() == 8`, a loop of 32 items gets budget 1 (the outer loop already
saturates the machine) while a loop of 2 items gets budget 4 (2 workers × 4 inner threads).
Iterators of unknown length fall back to the conservative budget of 1.
"""
budget_for(range) = _budget_for(Base.IteratorSize(range), range)
_budget_for(::Union{Base.HasLength, Base.HasShape}, range) =
    max(1, capacity() ÷ max(1, length(range)))
_budget_for(::Base.IteratorSize, _) = 1
