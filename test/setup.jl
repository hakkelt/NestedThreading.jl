"""
Synthetic pools that make the applied budget observable without depending on how a real
library clamps thread counts.

Registered once per test process (the module body runs once; registering a name twice
throws). Both pools stay registered for the whole session on
purpose — that is exactly the state a downstream package would see.
"""
@testmodule Probe begin
    using NestedThreading
    using Test

    const NT = NestedThreading

    # Counted pool: records the exact budget applied.
    const VALUE = Ref(16)
    const APPLIED = Ref{Vector{Int}}(Int[])

    # Guarded pool: records the budget it was invoked with, and its nesting depth.
    const DEPTH = Threads.Atomic{Int}(0)
    const BUDGETS = Ref{Vector{Int}}(Int[])
    const LOCK = ReentrantLock()

    # Quiesce pool: counts how often it was asked to release Julia's threads.
    const QUIESCED = Threads.Atomic{Int}(0)

    # Park hook of the counted pool: counts how often it ran.
    const PARKED = Threads.Atomic{Int}(0)

    function guard(f, budget)
        @lock LOCK push!(BUDGETS[], budget)
        Threads.atomic_add!(DEPTH, 1)
        try
            return f()
        finally
            Threads.atomic_sub!(DEPTH, 1)
        end
    end

    function register!()
        NT.register_pool!(
            :probe;
            get = () -> VALUE[],
            set = n -> (VALUE[] = n; @lock LOCK push!(APPLIED[], n); n),
            park = () -> (Threads.atomic_add!(PARKED, 1); nothing),
        )
        NT.register_pool!(:probe_guard; guard)
        return NT.register_pool!(
            :probe_quiesce; quiesce = () -> (Threads.atomic_add!(QUIESCED, 1); nothing)
        )
    end

    register!()

    "Reset the probe to a known idle state. Call at the top of every test item."
    function reset!()
        VALUE[] = 16
        APPLIED[] = Int[]
        BUDGETS[] = Int[]
        QUIESCED[] = 0
        PARKED[] = 0
        return nothing
    end

    "How often the counted pool's park hook ran since the last `reset!`."
    park_count() = PARKED[]

    "How often the quiesce pool was asked to release Julia's threads since the last `reset!`."
    quiesce_count() = QUIESCED[]

    "Index of the probe pool in the parallel `MAXIMA`/`SAVED` arrays."
    probe_index() = findfirst(p -> p.name === :probe, NT.COUNTED_POOLS)

    "The budget the guard was last invoked with, or `nothing` if it was never invoked."
    last_budget() = isempty(BUDGETS[]) ? nothing : BUDGETS[][end]

    "Whether the guard ran at all since the last `reset!`."
    guarded() = !isempty(BUDGETS[])
end
