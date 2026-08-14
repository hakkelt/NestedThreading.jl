# A synthetic pool makes the applied budget observable without depending on how a real
# library clamps counts. It stays registered for the whole test session on purpose.
const PROBE = Ref(16)
const PROBE_LOG = Ref{Vector{Int}}(Int[])

NT.register_counted_pool!(
    () -> PROBE[],
    n -> (PROBE[] = n; push!(PROBE_LOG[], n); n);
    name = :probe,
)

# Guarded pools have no imperative state; record the flag they were invoked with.
const GUARD_DEPTH = Threads.Atomic{Int}(0)
const GUARD_FLAGS = Ref{Vector{Bool}}(Bool[])
const PROBE_LOCK = ReentrantLock()

NT.register_guarded_pool!(name = :probe_guard) do f, restricted
    @lock PROBE_LOCK push!(GUARD_FLAGS[], restricted)
    Threads.atomic_add!(GUARD_DEPTH, 1)
    try
        f()
    finally
        Threads.atomic_sub!(GUARD_DEPTH, 1)
    end
end

reset_probe!() = (PROBE[] = 16; PROBE_LOG[] = Int[]; GUARD_FLAGS[] = Bool[])

@testset "registration" begin
    @test :blas in [p.name for p in NT.COUNTED_POOLS]
    @test :probe in [p.name for p in NT.COUNTED_POOLS]
    @test :probe_guard in [p.name for p in NT.GUARDED_POOLS]
    @test length(NT.SAVED) == length(NT.COUNTED_POOLS)
    @test length(NT.MAXIMA) == length(NT.COUNTED_POOLS)

    # Registering the same name twice is a no-op rather than a duplicate pool.
    before = length(NT.COUNTED_POOLS)
    NT.register_counted_pool!(() -> 1, identity; name = :probe)
    @test length(NT.COUNTED_POOLS) == before
end

@testset "scoped save/apply/restore" begin
    reset_probe!()
    original = BLAS.get_num_threads()

    result = NT.with_thread_budget(3) do
        @test PROBE[] == 3
        @test BLAS.get_num_threads() == 3
        :done
    end
    @test result === :done
    @test PROBE[] == 16
    @test BLAS.get_num_threads() == original

    # Budgets are clamped to >= 1.
    NT.with_thread_budget(0) do
        @test PROBE[] == 1
    end
    NT.with_thread_budget(-5) do
        @test PROBE[] == 1
    end

    # The restore happens on the exception path too.
    @test_throws ErrorException NT.with_thread_budget(() -> error("boom"), 2)
    @test PROBE[] == 16
    @test BLAS.get_num_threads() == original
    @test isempty(NT.ACTIVE)
end

@testset "nesting only ever narrows" begin
    reset_probe!()
    NT.with_thread_budget(2) do
        @test PROBE[] == 2
        NT.with_thread_budget(8) do          # asks for more...
            @test PROBE[] == 2               # ...and does not get it
        end
        @test PROBE[] == 2
        NT.with_restricted_threads() do
            @test PROBE[] == 1
        end
        @test PROBE[] == 2                   # inner exit restores the outer budget,
    end                                      # not the pre-scope value
    @test PROBE[] == 16
end

@testset "with_full_threads / with_restricted_threads" begin
    reset_probe!()
    NT.with_full_threads() do
        @test PROBE[] == NT.capacity()
        @test GUARD_FLAGS[][end] == false     # guards actively enabled
    end
    NT.with_restricted_threads() do
        @test PROBE[] == 1
        @test GUARD_FLAGS[][end] == true
    end
    # Nested inside a restriction, a full-threads request is still clamped.
    NT.with_restricted_threads() do
        NT.with_full_threads() do
            @test PROBE[] == 1
        end
    end
    @test PROBE[] == 16
end

@testset "guarded pools follow the restriction, not the exact count" begin
    reset_probe!()
    if NT.capacity() > 1
        NT.with_thread_budget(NT.capacity() - 1) do
            # A partial budget must still disable pools that cannot be partially limited.
            @test GUARD_FLAGS[][end] == true
        end
    end
    NT.with_thread_budget(NT.capacity()) do
        @test GUARD_FLAGS[][end] == false
    end
    @test GUARD_DEPTH[] == 0
end

@testset "exclude skips a named guard" begin
    reset_probe!()
    NT.with_thread_budget(1; exclude = (:probe_guard,)) do
        @test GUARD_DEPTH[] == 0
        @test isempty(GUARD_FLAGS[])
        @test PROBE[] == 1                    # counted pools are still applied
    end
end

@testset "budget_for" begin
    n = NT.capacity()
    @test NT.budget_for(1:1) == n
    @test NT.budget_for(1:(2n)) == 1
    @test NT.budget_for(1:n) == 1
    @test NT.budget_for(Int[]) == n           # empty range: no division by zero
    @test NT.budget_for(CartesianIndices((4, 4))) == max(1, n ÷ 16)
    @test NT.budget_for(Iterators.filter(isodd, 1:10)) == 1   # unknown length -> conservative
end

@testset "enable_full_threading sets the baseline" begin
    reset_probe!()
    saved_blas = BLAS.get_num_threads()
    try
        PROBE[] = 2
        NT.enable_full_threading()
        @test PROBE[] == NT.MAXIMA[findfirst(p -> p.name === :probe, NT.COUNTED_POOLS)]

        # An active scope still wins while it is active, and the *baseline* is what the
        # last exit restores to.
        PROBE[] = 2
        NT.with_thread_budget(1) do
            @test PROBE[] == 1
            NT.enable_full_threading()
            @test PROBE[] == 1
        end
        @test PROBE[] == NT.MAXIMA[findfirst(p -> p.name === :probe, NT.COUNTED_POOLS)]
    finally
        BLAS.set_num_threads(saved_blas)
    end
end

@testset "concurrent scopes never corrupt the bookkeeping" begin
    # This is the regression test for the save/restore race: with per-scope save/restore
    # (even under a lock) tasks capture each other's already-restricted counts and the
    # process ends up permanently throttled.
    reset_probe!()
    original = BLAS.get_num_threads()
    PROBE[] = original

    violations = Threads.Atomic{Int}(0)
    ntasks = 32
    tasks = map(1:ntasks) do i
        Threads.@spawn begin
            for _ in 1:50
                budget = 1 + (i % 4)
                NT.with_thread_budget(budget) do
                    # Invariant: while this scope is active nobody may have raised the
                    # applied count above what we asked for.
                    PROBE[] > budget && Threads.atomic_add!(violations, 1)
                    yield()
                    PROBE[] > budget && Threads.atomic_add!(violations, 1)
                end
            end
        end
    end
    foreach(wait, tasks)

    @test violations[] == 0
    @test isempty(NT.ACTIVE)
    @test PROBE[] == original
    @test BLAS.get_num_threads() == original
end
