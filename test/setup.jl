"""
Synthetic pools that make the applied budget observable without depending on how a real
library clamps thread counts.

Registered once per test process (the module body runs once, and registration is a no-op
for a name that already exists). Both pools stay registered for the whole session on
purpose — that is exactly the state a downstream package would see.
"""
@testmodule Probe begin
    using NestedThreading
    using Test

    const NT = NestedThreading

    # Counted pool: records the exact budget applied.
    const VALUE = Ref(16)
    const APPLIED = Ref{Vector{Int}}(Int[])

    # Guarded pool: records the `restricted` flag it was invoked with, and its nesting depth.
    const DEPTH = Threads.Atomic{Int}(0)
    const FLAGS = Ref{Vector{Bool}}(Bool[])
    const LOCK = ReentrantLock()

    function register!()
        NT.register_counted_pool!(
            () -> VALUE[],
            n -> (VALUE[] = n; @lock LOCK push!(APPLIED[], n); n);
            name = :probe,
        )
        return NT.register_guarded_pool!(name = :probe_guard) do f, restricted
            @lock LOCK push!(FLAGS[], restricted)
            Threads.atomic_add!(DEPTH, 1)
            try
                f()
            finally
                Threads.atomic_sub!(DEPTH, 1)
            end
        end
    end

    register!()

    "Reset the probe to a known idle state. Call at the top of every test item."
    function reset!()
        VALUE[] = 16
        APPLIED[] = Int[]
        FLAGS[] = Bool[]
        return nothing
    end

    "Index of the probe pool in the parallel `MAXIMA`/`SAVED` arrays."
    probe_index() = findfirst(p -> p.name === :probe, NT.COUNTED_POOLS)

    "The `restricted` flag the guard was last invoked with."
    last_flag() = FLAGS[][end]
end
