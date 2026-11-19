@testitem "public names" tags = [:registry] begin
    using NestedThreading
    if VERSION >= v"1.11"
        for name in (
                :register_pool!, :capacity, :thread_limit, :quiesce_foreign_pools,
                :reset_thread_counts!,
            )
            @test Base.ispublic(NestedThreading, name)
        end
        for name in (
                :CountedPool, :GuardedPool, :QuiescePool, :park_openblas, :budget_for,
            )
            @test !Base.ispublic(NestedThreading, name)
        end
    end
    # Exported: the loop macros and the scope functions, nothing else.
    @test Set(filter(n -> Base.isexported(NestedThreading, n), names(NestedThreading))) == Set(
        [
            :NestedThreading, Symbol("@budgeted"), Symbol("@budgeted_threads"),
            Symbol("@budgeted_batch"), :with_threads_disabled, :with_threads_enabled,
            :with_thread_limit, :with_thread_default, :with_thread_grant, :thread_pool_info,
        ]
    )
end

@testitem "pool registration" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading

    @test :blas in [p.name for p in NT.COUNTED_POOLS]
    @test :probe in [p.name for p in NT.COUNTED_POOLS]
    @test :probe_guard in [p.name for p in NT.GUARDED_POOLS]
    @test :probe_quiesce in [p.name for p in NT.QUIESCE_POOLS]
    @test length(NT.SAVED) == length(NT.COUNTED_POOLS)
    @test length(NT.MAXIMA) == length(NT.COUNTED_POOLS)

    # Every rejected registration leaves the registry exactly as it was.
    state() = (
        copy(NT.POOL_NAMES[]), length(NT.COUNTED_POOLS), length(NT.GUARDED_POOLS),
        length(NT.QUIESCE_POOLS), length(NT.SAVED), length(NT.MAXIMA),
    )
    before = state()
    g() = 1
    s(n) = n
    @test_throws "a pool named :probe is already registered" NT.register_pool!(
        :probe; get = g, set = s
    )
    @test_throws "a pool named :probe_guard is already registered" NT.register_pool!(
        :probe_guard; quiesce = () -> nothing
    )
    @test_throws "has `get` but no `set`" NT.register_pool!(:bad_probe; get = g)
    @test_throws "has `set` but no `get`" NT.register_pool!(:bad_probe; set = s)
    @test_throws "has `park` but no `get`/`set`" NT.register_pool!(
        :bad_probe; guard = (f, n) -> f(), park = () -> nothing
    )
    @test_throws "has no control" NT.register_pool!(:bad_probe)
    @test_throws ArgumentError NT.register_pool!(:bad_probe)
    @test_throws "returned 2.0 of type Float64" NT.register_pool!(
        :bad_probe; get = () -> 2.0, set = s
    )
    @test_throws "returned nothing of type Nothing" NT.register_pool!(
        :bad_probe; get = () -> nothing, set = s
    )
    @test state() == before
    @test !(:bad_probe in NT.POOL_NAMES[])
end

@testitem "thread_pool_info describes the registry" tags = [:registry] setup = [Probe] begin
    using NestedThreading, LinearAlgebra
    const NT = NestedThreading
    Probe.reset!()

    info = thread_pool_info()
    @test info isa Vector{<:NamedTuple}
    @test [p.name for p in info] == NT.POOL_NAMES[]
    byname(name, info = thread_pool_info()) = only(filter(p -> p.name === name, info))

    blas = byname(:blas, info)
    @test (blas.has_count, blas.has_guard, blas.has_quiesce, blas.has_park) ==
        (true, false, false, true)
    @test blas.current == blas.unrestricted == blas.effective == BLAS.get_num_threads()

    probe = byname(:probe, info)
    @test (probe.has_count, probe.has_guard, probe.has_quiesce, probe.has_park) ==
        (true, false, false, true)
    @test probe.current == probe.unrestricted == probe.effective == 16

    guard = byname(:probe_guard, info)
    @test (guard.has_count, guard.has_guard, guard.has_quiesce, guard.has_park) ==
        (false, true, false, false)
    @test guard.current === guard.unrestricted === guard.effective === nothing
    @test byname(:probe_quiesce, info).has_quiesce

    # Inside a scope: `current` and `effective` follow the limit, `unrestricted` is the
    # value the last scope restores.
    with_thread_limit(3; only = (:probe,)) do
        probe = byname(:probe)
        @test (probe.current, probe.unrestricted, probe.effective) == (3, 16, 3)
        @test byname(:blas).effective == BLAS.get_num_threads()
    end
    @test byname(:probe).current == 16
end

@testitem "quiesce_foreign_pools calls every registered hook once" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    NT.quiesce_foreign_pools()
    @test Probe.quiesce_count() == 1
    NT.quiesce_foreign_pools()
    @test Probe.quiesce_count() == 2
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
    NT.with_thread_limit(1) do
        NT.register_pool!(:late_probe; get = () -> late[], set = n -> (late[] = n))
        @test late[] == 1                       # the restriction reached the newcomer
    end
    @test late[] == 12                          # ...and its own value came back

    i = findfirst(p -> p.name === :late_probe, NT.COUNTED_POOLS)
    @test i !== nothing
    @test NT.MAXIMA[i] == 12
    @test length(NT.SAVED) == length(NT.MAXIMA) == length(NT.COUNTED_POOLS)

    # It now participates in ordinary scopes like any other pool.
    NT.with_thread_limit(3) do
        @test late[] == 3
    end
    @test late[] == 12

    # A guard registered inside an open scope does not apply to that scope, and the scope
    # still runs to completion; the next restricted scope invokes it. `const` for the same
    # reason as `late`.
    const late_guarded = Threads.Atomic{Int}(0)
    result = NT.with_thread_limit(1) do
        NT.register_pool!(
            :late_guard; guard = (f, n) -> (Threads.atomic_add!(late_guarded, 1); f())
        )
        :body_done
    end
    @test result === :body_done
    @test late_guarded[] == 0
    NT.with_thread_limit(1) do
        nothing
    end
    @test late_guarded[] == 1
end

@testitem "scoped save/apply/restore" tags = [:registry] setup = [Probe] begin
    using NestedThreading, LinearAlgebra
    const NT = NestedThreading
    Probe.reset!()

    original = BLAS.get_num_threads()

    result = NT.with_thread_limit(3) do
        @test Probe.VALUE[] == 3
        @test BLAS.get_num_threads() == 3
        :done
    end
    @test result === :done
    @test Probe.VALUE[] == 16
    @test BLAS.get_num_threads() == original

    # Budgets are clamped to >= 1.
    NT.with_thread_limit(0) do
        @test Probe.VALUE[] == 1
    end
    NT.with_thread_limit(-5) do
        @test Probe.VALUE[] == 1
    end

    # The restore happens on the exception path too.
    @test_throws ErrorException NT.with_thread_limit(() -> error("boom"), 2)
    @test Probe.VALUE[] == 16
    @test BLAS.get_num_threads() == original
    @test isempty(NT.ACTIVE)
end

@testitem "nesting only ever narrows" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    NT.with_thread_limit(2) do
        @test Probe.VALUE[] == 2
        NT.with_thread_limit(8) do          # asks for more...
            @test Probe.VALUE[] == 2         # ...and does not get it
        end
        @test Probe.VALUE[] == 2
        NT.with_threads_disabled() do
            @test Probe.VALUE[] == 1
        end
        @test Probe.VALUE[] == 2             # inner exit restores the outer budget,
    end                                      # not the pre-scope value
    @test Probe.VALUE[] == 16
end

@testitem "with_threads_enabled / with_threads_disabled" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    NT.with_threads_enabled() do
        @test Probe.VALUE[] == NT.capacity()
        # Nothing is restricted, so guards are skipped entirely rather than invoked with a
        # "not restricted" flag.
        @test !Probe.guarded()
    end
    Probe.reset!()
    NT.with_threads_disabled() do
        @test Probe.VALUE[] == 1
        NT.capacity() > 1 && @test Probe.last_budget() == 1
    end
    # Nested inside a restriction, a full-threads request is still clamped.
    NT.with_threads_disabled() do
        NT.with_threads_enabled() do
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
        NT.with_thread_limit(NT.capacity() - 1) do
            @test Probe.last_budget() == NT.capacity() - 1
        end
        Probe.reset!()
        NT.with_threads_disabled() do
            @test Probe.last_budget() == 1
        end
        Probe.reset!()
    end

    # An unrestricted scope skips guards altogether.
    NT.with_thread_limit(NT.capacity()) do
        @test !Probe.guarded()
    end
    @test Probe.DEPTH[] == 0
end

@testitem "exclude skips a named guard" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    NT.with_thread_limit(1; exclude=(:probe_guard,)) do
        @test Probe.DEPTH[] == 0
        @test !Probe.guarded()
        @test Probe.VALUE[] == 1              # counted pools are still applied
    end

    # A name that matches no registered guard is simply ignored, not an error — a caller
    # may exclude a pool whose package is not loaded in this session.
    Probe.reset!()
    NT.with_thread_limit(1; exclude=(:not_a_pool,)) do
        NT.capacity() > 1 && @test Probe.guarded()
        @test Probe.VALUE[] == 1
    end
end

@testitem "exclude leaves a counted pool alone" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    # An excluded thread count keeps its pre-scope value, not merely unrestricted by this
    # scope but untouched, while the scope's guards still apply.
    Probe.VALUE[] = 7
    NT.with_thread_limit(1; exclude=(:probe,)) do
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
    NT.with_thread_limit(1; only=(:blas,)) do
        @test BLAS.get_num_threads() == 1
        @test Probe.VALUE[] == 7
        NT.capacity() > 1 && @test !Probe.guarded()
    end
    @test BLAS.get_num_threads() == original_blas
    @test Probe.VALUE[] == 7

    # `only = (:probe,)` is the mirror image: the probe counted pool is restricted, BLAS and
    # the probe guard are not.
    Probe.reset!()
    NT.with_thread_limit(1; only=(:probe,)) do
        @test Probe.VALUE[] == 1
        @test BLAS.get_num_threads() == original_blas
        @test !Probe.guarded()
    end
    @test Probe.VALUE[] == 16

    # Nesting still only narrows: an outer plain restriction still wins even where an inner
    # scope's `only` does not name the pool.
    Probe.reset!()
    NT.with_thread_limit(1) do
        NT.with_thread_limit(NT.capacity(); only=(:blas,)) do
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
    # caller configured by hand. That is what lets `with_threads_enabled` turn a default-off
    # library (NFFT) on; the hand-set value is restored on exit.
    if NT.capacity() > 1
        Probe.VALUE[] = 1
        NT.with_threads_enabled() do
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

@testitem "reset_thread_counts! sets the baseline" tags = [:registry] setup = [Probe] begin
    using NestedThreading, LinearAlgebra
    const NT = NestedThreading
    Probe.reset!()

    saved_blas = BLAS.get_num_threads()
    maximum_probe = NT.MAXIMA[Probe.probe_index()]
    try
        Probe.VALUE[] = 2
        NT.reset_thread_counts!()
        @test Probe.VALUE[] == maximum_probe

        # An active scope still wins while it is active, and the *baseline* is what the
        # last exit restores to.
        Probe.VALUE[] = 2
        NT.with_thread_limit(1) do
            @test Probe.VALUE[] == 1
            NT.reset_thread_counts!()
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
                NT.with_thread_limit(budget) do
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

@testitem "soft defaults yield to grants, grants yield to limits" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    NT.with_thread_default(1) do
        @test Probe.VALUE[] == 1
        NT.with_thread_grant(8) do
            @test Probe.VALUE[] == 8                  # a grant overrides the soft default
        end
        @test Probe.VALUE[] == 1
        NT.with_thread_limit(4) do                   # a limit is a ceiling, not a request
            @test Probe.VALUE[] == 1
            NT.with_thread_grant(8) do
                @test Probe.VALUE[] == 4              # ...and a grant never passes it
            end
        end
        NT.with_thread_limit(1) do
            NT.with_thread_grant(8) do
                @test Probe.VALUE[] == 1
            end
        end
    end
    @test Probe.VALUE[] == 16

    # A limit opened around the soft default clamps it the same way.
    NT.with_thread_limit(2) do
        NT.with_thread_default(6) do
            @test Probe.VALUE[] == 2
            NT.with_thread_grant(8) do
                @test Probe.VALUE[] == 2
            end
        end
    end

    # With no soft default open, a grant neither lowers a pool nor raises it past the
    # unrestricted value.
    NT.with_thread_grant(8) do
        @test Probe.VALUE[] == 16
    end
    Probe.VALUE[] = 3
    NT.with_thread_grant(8) do
        @test Probe.VALUE[] == 3
    end
    @test Probe.VALUE[] == 3
    Probe.reset!()
    @test isempty(NT.ACTIVE)
end

@testitem "soft defaults combine by minimum, grants by maximum" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    NT.with_thread_default(4) do
        NT.with_thread_default(2) do
            @test Probe.VALUE[] == 2
            NT.with_thread_default(6) do
                @test Probe.VALUE[] == 2
            end
            NT.with_thread_grant(5) do
                @test Probe.VALUE[] == 5
                NT.with_thread_grant(3) do
                    @test Probe.VALUE[] == 5
                end
                NT.with_thread_grant(9) do
                    @test Probe.VALUE[] == 9
                end
                @test Probe.VALUE[] == 5
            end
        end
        @test Probe.VALUE[] == 4
    end
    @test Probe.VALUE[] == 16

    # `only` scopes a soft default like any other entry.
    NT.with_thread_default(1; only = (:blas,)) do
        @test Probe.VALUE[] == 16
    end
end

@testitem "park hooks run when a grant lowers their pool" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    NT.with_thread_default(1) do
        NT.with_thread_grant(8) do
            @test Probe.park_count() == 0
        end
        @test Probe.park_count() == 1               # 8 -> 1

        # A grant that raised nothing parks nothing.
        NT.with_thread_limit(1) do
            NT.with_thread_grant(Returns(nothing), 8)
        end
        @test Probe.park_count() == 1

        # Nor does the inner of two grants, while the outer keeps the pool up.
        NT.with_thread_grant(8) do
            NT.with_thread_grant(Returns(nothing), 8)
            @test Probe.park_count() == 1
        end
        @test Probe.park_count() == 2
    end

    # Limits and soft defaults never park, even when their exit lowers nothing or restores.
    NT.with_thread_limit(Returns(nothing), 2)
    NT.with_thread_default(Returns(nothing), 2)
    @test Probe.park_count() == 2

    # A grant that is the last scope open parks only if the restore lowers the pool; with no
    # soft default around it, it raised nothing.
    NT.with_thread_grant(Returns(nothing), 8)
    @test Probe.park_count() == 2

    # Inside a partial hard limit -- a parallel loop whose workers may be inside the library --
    # nothing is parked, though the grant still takes effect. A full-capacity limit, which a
    # loop of one item opens, does not count.
    if NT.capacity() > 2
        NT.with_thread_default(1) do
            NT.with_thread_limit(2) do
                NT.with_thread_grant(8) do
                    @test Probe.VALUE[] == 2
                end
            end
            @test Probe.park_count() == 2
            NT.with_thread_limit(NT.capacity()) do
                NT.with_thread_grant(Returns(nothing), 8)
            end
            @test Probe.park_count() == 3
        end
    end
    Probe.reset!()
end

@testitem "concurrent defaults, grants and limits keep the invariants" tags = [:registry] setup = [Probe] begin
    using NestedThreading, LinearAlgebra
    const NT = NestedThreading
    Probe.reset!()

    original = BLAS.get_num_threads()
    violations = Threads.Atomic{Int}(0)
    tasks = map(1:24) do i
        Threads.@spawn begin
            for _ in 1:40
                limit = 1 + (i % 3)
                NT.with_thread_limit(limit) do
                    NT.with_thread_default(1) do
                        NT.with_thread_grant(8) do
                            # Whatever else is open, nothing may exceed this task's limit.
                            Probe.VALUE[] > limit && Threads.atomic_add!(violations, 1)
                            yield()
                            Probe.VALUE[] > limit && Threads.atomic_add!(violations, 1)
                        end
                        Probe.VALUE[] > limit && Threads.atomic_add!(violations, 1)
                    end
                end
            end
        end
    end
    foreach(wait, tasks)

    @test violations[] == 0
    @test isempty(NT.ACTIVE)
    @test Probe.VALUE[] == 16
    @test BLAS.get_num_threads() == original
end

@testitem "the OpenBLAS park hook leaves BLAS usable" tags = [:registry] begin
    using NestedThreading, LinearAlgebra
    const NT = NestedThreading

    @test any(p -> p.name === :blas && p.park === NT.park_openblas, NT.COUNTED_POOLS)
    A = randn(256, 256)
    expected = A * A
    original = BLAS.get_num_threads()
    NT.with_thread_default(1; only = (:blas,)) do
        for _ in 1:3
            NT.with_thread_grant(NT.capacity(); only = (:blas,)) do
                @test A * A ≈ expected
            end
            @test BLAS.get_num_threads() == 1
            @test A * A ≈ expected                      # after the workers were shut down
        end
    end
    @test BLAS.get_num_threads() == original
    NT.park_openblas()                                  # also callable with nothing open
    @test A * A ≈ expected
end

@testitem "thread_limit reports the enclosing hard limit" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading

    @test NT.thread_limit(:probe) === nothing
    if VERSION < v"1.11"
        # No `ScopedValue`: enclosing scopes are not tracked, so nothing is reported.
        with_threads_disabled() do
            @test NT.thread_limit(:probe) === nothing
        end
    else
        with_threads_disabled() do
            @test NT.thread_limit(:probe) == 1
            @test NT.thread_limit(:blas) == 1
            # An unregistered name reports what a pool of that name would be held to.
            @test NT.thread_limit(:not_registered) == 1
            # A task spawned inside the scope runs inside it too.
            @test fetch(Threads.@spawn NT.thread_limit(:probe)) == 1
        end
        @test NT.thread_limit(:probe) === nothing

        # Nested limits combine by minimum, and the outer one survives the inner one's exit.
        NT.with_thread_limit(3) do
            @test NT.thread_limit(:probe) == 3
            NT.with_thread_limit(2) do
                @test NT.thread_limit(:probe) == 2
            end
            @test NT.thread_limit(:probe) == 3
        end

        # `exclude`/`only` decide which pools a limit applies to.
        NT.with_thread_limit(1; exclude = (:probe,)) do
            @test NT.thread_limit(:probe) === nothing
            @test NT.thread_limit(:blas) == 1
        end
        NT.with_thread_limit(1; only = (:probe,)) do
            @test NT.thread_limit(:probe) == 1
            @test NT.thread_limit(:blas) === nothing
        end

        # Soft defaults and grants bound nothing, so they do not count.
        NT.with_thread_default(1) do
            @test NT.thread_limit(:probe) === nothing
            NT.with_thread_grant(4) do
                @test NT.thread_limit(:probe) === nothing
            end
        end
        NT.with_thread_limit(2) do
            NT.with_thread_grant(4) do
                @test NT.thread_limit(:probe) == 2
            end
        end

        # A limit another task holds is not this task's: it may close at any moment.
        opened, release = Channel{Nothing}(1), Channel{Nothing}(1)
        holder = Threads.@spawn with_threads_disabled() do
            put!(opened, nothing)
            take!(release)
        end
        take!(opened)
        @test NT.thread_limit(:probe) === nothing
        put!(release, nothing)
        wait(holder)
    end
    @test isempty(NT.ACTIVE)
end

@testitem "a nested limit that could change nothing is not entered" tags = [:registry] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading

    Probe.reset!()
    # On 1.10 every scope is entered; the fast path needs `ScopedValue`.
    skips = VERSION >= v"1.11"
    with_threads_disabled() do
        applied = copy(Probe.APPLIED[])
        guarded = copy(Probe.BUDGETS[])
        active = length(NT.ACTIVE)
        # Same or larger budget, every pool already held at 1 by this enclosing scope: a
        # no-op, so nothing is entered, no counted pool is set and no guard runs again.
        @test with_threads_disabled(() -> length(NT.ACTIVE)) == active + !skips
        @test NT.with_thread_limit(() -> length(NT.ACTIVE), 3) == active + !skips
        # ...and the same inside a task spawned in the scope, a batch-loop worker's case.
        @test fetch(Threads.@spawn with_threads_disabled(() -> length(NT.ACTIVE))) == active + !skips
        if skips
            @test Probe.APPLIED[] == applied
            @test Probe.BUDGETS[] == guarded
        end
        @test Probe.VALUE[] == 1
        # The body's value and exceptions still come through.
        @test with_threads_disabled(() -> 42) == 42
        @test_throws ErrorException with_threads_disabled(() -> error("boom"))
        @test length(NT.ACTIVE) == active
    end
    @test isempty(NT.ACTIVE)
    @test Probe.VALUE[] == 16

    # A smaller budget is entered.
    NT.with_thread_limit(4) do
        active = length(NT.ACTIVE)
        NT.with_thread_limit(2) do
            @test length(NT.ACTIVE) == active + 1
            @test Probe.VALUE[] == 2
        end
        @test Probe.VALUE[] == 4
    end

    # An outer limit that leaves a pool out does not make the inner one a no-op.
    NT.with_thread_limit(1; exclude = (:probe,)) do
        active = length(NT.ACTIVE)
        with_threads_disabled() do
            @test length(NT.ACTIVE) == active + 1
            @test Probe.VALUE[] == 1
        end
        @test Probe.VALUE[] == 16
    end
    # ...but an inner one confined to the pools the outer covers is.
    NT.with_thread_limit(1; only = (:probe,)) do
        active = length(NT.ACTIVE)
        NT.with_thread_limit(1; only = (:probe,)) do
            @test length(NT.ACTIVE) == active + !skips
        end
    end

    # A soft default around the call is not a hard limit, so the inner limit is entered.
    NT.with_thread_default(1) do
        active = length(NT.ACTIVE)
        with_threads_disabled() do
            @test length(NT.ACTIVE) == active + 1
        end
    end

    # A limit an unrelated task holds does not make this task's limit a no-op: it is entered,
    # and stays in force after the other task's scope closes.
    opened, release = Channel{Nothing}(1), Channel{Nothing}(1)
    holder = Threads.@spawn with_threads_disabled() do
        put!(opened, nothing)
        take!(release)
    end
    take!(opened)
    with_threads_disabled() do
        @test length(NT.ACTIVE) == 2
        put!(release, nothing)
        wait(holder)
        @test Probe.VALUE[] == 1
    end
    @test isempty(NT.ACTIVE)
    @test Probe.VALUE[] == 16
end

@testitem "quiet skips are logged at debug level" tags = [:registry] setup = [Probe] begin
    using NestedThreading, Logging
    const NT = NestedThreading
    Probe.reset!()
    const D = Logging.Debug
    noop() = nothing

    # Unknown names stay legal (the library may register later) but are reported.
    @test_logs (:debug, r"No pool named :no_such_pool is registered") min_level = D match_mode = :any with_thread_limit(
        noop, 2; exclude = (:no_such_pool,)
    )
    @test_logs (:debug, r"No pool named :no_such_pool is registered") min_level = D match_mode = :any with_thread_default(
        noop, 2; only = (:probe, :no_such_pool)
    )
    @test_logs (:debug, r"thread_limit: no pool named :no_such_pool") min_level = D NT.thread_limit(
        :no_such_pool
    )
    # Registered names, and a count of at least 1, are not reported.
    @test_logs min_level = D with_thread_limit(noop, 2; exclude = (:probe,))
    @test_logs min_level = D NT.thread_limit(:probe)

    @test_logs (:debug, r"Thread count 0 raised to 1") min_level = D match_mode = :any with_thread_limit(
        noop, 0
    )

    # A grant that lowers a pool under a partial hard limit does not park it.
    if NT.capacity() > 2
        with_thread_limit(2) do
            with_thread_default(1; only = (:probe,)) do
                @test_logs (:debug, r"Pool :probe not parked") min_level = D match_mode = :any with_thread_grant(
                    noop, NT.capacity(); only = (:probe,)
                )
            end
        end
        @test Probe.park_count() == 0
    end

    # A guard registered inside an open scope applies only to later scopes.
    @test_logs (:debug, r"Guard of pool :debug_guard registered while a scope is open") min_level = D match_mode = :any with_thread_limit(
        1
    ) do
        NT.register_pool!(:debug_guard; guard = (f, n) -> f())
    end
end
