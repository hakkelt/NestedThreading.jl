@testitem "pool registration" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading

    @test :blas in [p.name for p in NT.COUNTED_POOLS]
    @test :probe in [p.name for p in NT.COUNTED_POOLS]
    @test :probe_guard in [p.name for p in NT.GUARDED_POOLS]
    @test length(NT.SAVED) == length(NT.COUNTED_POOLS)
    @test length(NT.MAXIMA) == length(NT.COUNTED_POOLS)

    # Registering the same name twice is a no-op rather than a duplicate pool.
    before = length(NT.COUNTED_POOLS)
    NT.register_counted_pool!(() -> 1, identity; name=:probe)
    @test length(NT.COUNTED_POOLS) == before
    @test count(p -> p.name === :probe, NT.COUNTED_POOLS) == 1

    # ...and the same for guarded pools.
    before_guarded = length(NT.GUARDED_POOLS)
    NT.register_guarded_pool!((f, budget) -> f(); name=:probe_guard)
    @test length(NT.GUARDED_POOLS) == before_guarded
    @test count(p -> p.name === :probe_guard, NT.GUARDED_POOLS) == 1

    # The three parallel arrays stay index-aligned after all of that.
    @test length(NT.SAVED) == length(NT.MAXIMA) == length(NT.COUNTED_POOLS)
end

@testitem "registering mid-scope joins the active restriction" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    # Registration is meant to happen at load time, but an extension can be loaded while
    # some other task holds a budget scope open. The newcomer must pick up the restriction
    # in force and be restored with everybody else, not be left throttled forever.
    #
    # `const`, not a plain global: `late_probe` stays registered in the global registry for
    # the rest of the process, so its getter/setter closures must go on seeing a live `late`
    # long after this test item itself has finished. TestItemRunner frees a plain global
    # (not a `const` one) right after the item completes, which would otherwise turn every
    # later `late[]` into a `MethodError` on `nothing`.
    const late = Ref(12)
    NT.with_thread_budget(1) do
        NT.register_counted_pool!(() -> late[], n -> (late[] = n); name=:late_probe)
        @test late[] == 1                       # the restriction reached the newcomer
    end
    @test late[] == 12                          # ...and its own value came back

    i = findfirst(p -> p.name === :late_probe, NT.COUNTED_POOLS)
    @test i !== nothing
    @test NT.MAXIMA[i] == 12
    @test length(NT.SAVED) == length(NT.MAXIMA) == length(NT.COUNTED_POOLS)

    # It now participates in ordinary scopes like any other pool.
    NT.with_thread_budget(3) do
        @test late[] == 3
    end
    @test late[] == 12
end

@testitem "scoped save/apply/restore" tags = [:registry] setup = [Probe] begin
    using NestedThreading, LinearAlgebra
    const NT = NestedThreading
    Probe.reset!()

    original = BLAS.get_num_threads()

    result = NT.with_thread_budget(3) do
        @test Probe.VALUE[] == 3
        @test BLAS.get_num_threads() == 3
        :done
    end
    @test result === :done
    @test Probe.VALUE[] == 16
    @test BLAS.get_num_threads() == original

    # Budgets are clamped to >= 1.
    NT.with_thread_budget(0) do
        @test Probe.VALUE[] == 1
    end
    NT.with_thread_budget(-5) do
        @test Probe.VALUE[] == 1
    end

    # The restore happens on the exception path too.
    @test_throws ErrorException NT.with_thread_budget(() -> error("boom"), 2)
    @test Probe.VALUE[] == 16
    @test BLAS.get_num_threads() == original
    @test isempty(NT.ACTIVE)
end

@testitem "nesting only ever narrows" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    NT.with_thread_budget(2) do
        @test Probe.VALUE[] == 2
        NT.with_thread_budget(8) do          # asks for more...
            @test Probe.VALUE[] == 2         # ...and does not get it
        end
        @test Probe.VALUE[] == 2
        NT.with_restricted_threads() do
            @test Probe.VALUE[] == 1
        end
        @test Probe.VALUE[] == 2             # inner exit restores the outer budget,
    end                                      # not the pre-scope value
    @test Probe.VALUE[] == 16
end

@testitem "with_full_threads / with_restricted_threads" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    NT.with_full_threads() do
        @test Probe.VALUE[] == NT.capacity()
        # Nothing is restricted, so guards are skipped entirely rather than invoked with a
        # "not restricted" flag.
        @test !Probe.guarded()
    end
    Probe.reset!()
    NT.with_restricted_threads() do
        @test Probe.VALUE[] == 1
        NT.capacity() > 1 && @test Probe.last_budget() == 1
    end
    # Nested inside a restriction, a full-threads request is still clamped.
    NT.with_restricted_threads() do
        NT.with_full_threads() do
            @test Probe.VALUE[] == 1
        end
    end
    @test Probe.VALUE[] == 16
end

@testitem "guarded pools receive the applied budget" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    if NT.capacity() > 1
        # The guard is handed the budget itself, so a library with a partial limit (as
        # Polyester has) can leave the outer loop's unused threads available instead of
        # switching itself off.
        NT.with_thread_budget(NT.capacity() - 1) do
            @test Probe.last_budget() == NT.capacity() - 1
        end
        Probe.reset!()
        NT.with_restricted_threads() do
            @test Probe.last_budget() == 1
        end
        Probe.reset!()
    end

    # An unrestricted scope skips guards altogether.
    NT.with_thread_budget(NT.capacity()) do
        @test !Probe.guarded()
    end
    @test Probe.DEPTH[] == 0
end

@testitem "exclude skips a named guard" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    NT.with_thread_budget(1; exclude=(:probe_guard,)) do
        @test Probe.DEPTH[] == 0
        @test !Probe.guarded()
        @test Probe.VALUE[] == 1              # counted pools are still applied
    end

    # A name that matches no registered guard is simply ignored, not an error — a caller
    # may exclude a pool whose package is not loaded in this session.
    Probe.reset!()
    NT.with_thread_budget(1; exclude=(:not_a_pool,)) do
        NT.capacity() > 1 && @test Probe.guarded()
        @test Probe.VALUE[] == 1
    end
end

@testitem "exclude now actually leaves a counted pool alone" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    # Regression test for C8: `exclude` used to be consulted only by `_run_guarded`, so
    # naming a counted pool did nothing — the pool was restricted anyway. Now it is left at
    # its pre-scope value, not merely unrestricted-by-this-scope but genuinely untouched.
    Probe.VALUE[] = 7
    NT.with_thread_budget(1; exclude=(:probe,)) do
        @test Probe.VALUE[] == 7
        NT.capacity() > 1 && @test Probe.guarded()   # the guard is unaffected
    end
    @test Probe.VALUE[] == 7
    Probe.reset!()
end

@testitem "only restricts exactly the named pools" tags = [:registry] setup = [Probe] begin
    using NestedThreading, LinearAlgebra
    const NT = NestedThreading
    Probe.reset!()

    original_blas = BLAS.get_num_threads()

    # `only = (:blas,)` must restrict BLAS and leave the probe pools — counted and guarded
    # alike — exactly as they were, even past this scope's own budget.
    Probe.VALUE[] = 7
    NT.with_thread_budget(1; only=(:blas,)) do
        @test BLAS.get_num_threads() == 1
        @test Probe.VALUE[] == 7
        NT.capacity() > 1 && @test !Probe.guarded()
    end
    @test BLAS.get_num_threads() == original_blas
    @test Probe.VALUE[] == 7

    # `only = (:probe,)` is the mirror image: the probe counted pool is restricted, BLAS and
    # the probe guard are not.
    Probe.reset!()
    NT.with_thread_budget(1; only=(:probe,)) do
        @test Probe.VALUE[] == 1
        @test BLAS.get_num_threads() == original_blas
        @test !Probe.guarded()
    end
    @test Probe.VALUE[] == 16

    # Nesting still only narrows: an outer plain restriction still wins even where an inner
    # scope's `only` does not name the pool.
    Probe.reset!()
    NT.with_thread_budget(1) do
        NT.with_thread_budget(NT.capacity(); only=(:blas,)) do
            @test Probe.VALUE[] == 1   # untouched by the inner scope, still bound by the outer
        end
    end
    Probe.reset!()
end

@testitem "a full budget raises past a hand-set count" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    # The applied budget is written to every counted pool unconditionally, so a scope whose
    # budget works out to capacity() sets the pools *up* to it, past a lower value the
    # caller configured by hand. That is what lets `with_full_threads` turn a default-off
    # library (NFFT) on; the hand-set value is restored on exit.
    if NT.capacity() > 1
        Probe.VALUE[] = 1
        NT.with_full_threads() do
            @test Probe.VALUE[] == NT.capacity()
        end
        @test Probe.VALUE[] == 1
    end
    Probe.reset!()
end

@testitem "budget_for" tags = [:registry] begin
    using NestedThreading
    const NT = NestedThreading

    n = NT.capacity()
    @test NT.budget_for(1:1) == n
    @test NT.budget_for(1:(2n)) == 1
    @test NT.budget_for(1:n) == 1
    @test NT.budget_for(Int[]) == n                             # empty: no division by zero
    @test NT.budget_for(CartesianIndices((4, 4))) == max(1, n ÷ 16)
    @test NT.budget_for(Iterators.filter(isodd, 1:10)) == 1      # unknown length: conservative
end

@testitem "enable_full_threading sets the baseline" tags = [:registry] setup = [Probe] begin
    using NestedThreading, LinearAlgebra
    const NT = NestedThreading
    Probe.reset!()

    saved_blas = BLAS.get_num_threads()
    maximum_probe = NT.MAXIMA[Probe.probe_index()]
    try
        Probe.VALUE[] = 2
        NT.enable_full_threading()
        @test Probe.VALUE[] == maximum_probe

        # An active scope still wins while it is active, and the *baseline* is what the
        # last exit restores to.
        Probe.VALUE[] = 2
        NT.with_thread_budget(1) do
            @test Probe.VALUE[] == 1
            NT.enable_full_threading()
            @test Probe.VALUE[] == 1
        end
        @test Probe.VALUE[] == maximum_probe
    finally
        BLAS.set_num_threads(saved_blas)
    end
end

@testitem "concurrent scopes never corrupt the bookkeeping" tags = [:registry] setup = [Probe] begin
    using NestedThreading, LinearAlgebra
    const NT = NestedThreading
    Probe.reset!()

    # Regression test for the save/restore race. With per-scope save/restore — even under a
    # lock — tasks capture each other's already-restricted counts and the process ends up
    # permanently throttled with no scope active.
    original = BLAS.get_num_threads()
    Probe.VALUE[] = original

    violations = Threads.Atomic{Int}(0)
    tasks = map(1:32) do i
        Threads.@spawn begin
            for _ in 1:50
                budget = 1 + (i % 4)
                NT.with_thread_budget(budget) do
                    # Invariant: while this scope is active nobody may have raised the
                    # applied count above what we asked for.
                    Probe.VALUE[] > budget && Threads.atomic_add!(violations, 1)
                    yield()
                    Probe.VALUE[] > budget && Threads.atomic_add!(violations, 1)
                end
            end
        end
    end
    foreach(wait, tasks)

    @test violations[] == 0
    @test isempty(NT.ACTIVE)
    @test Probe.VALUE[] == original
    @test BLAS.get_num_threads() == original
end
