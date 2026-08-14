using JET

# NOTE ON THE `broken` MARKERS BELOW
#
# `CountedPool.get`/`set` and `GuardedPool.guard` are abstractly-typed `Function` fields: a
# registry that downstream packages extend at load time cannot be concretely typed. The
# resulting dynamic dispatch is inherent to the design, not an oversight, and it is confined
# to `_apply!`/`_snapshot!`/`_restore!`/`_run_guarded`, which are `@noinline` precisely so it
# does not leak into callers' inlined code. It happens once per budget scope (once per
# `mul!`), never per loop iteration.
#
# What must stay clean is the *caller* side: the macros and the budget arithmetic.

@testset "JET: package" begin
    JET.test_package(NestedThreading; target_modules = (NestedThreading,))
end

@testset "JET: budget arithmetic is statically resolved" begin
    @test_opt NT.budget_for(1:8)
    @test_opt NT.budget_for(CartesianIndices((4, 4)))
    @test_opt NT.capacity()
    @test_call NT.budget_for(1:8)
end

@testset "JET: scoped entry points" begin
    f() = 1 + 1
    @test_call NT.with_thread_budget(f, 2)
    @test_call NT.with_restricted_threads(f)
    @test_call NT.with_full_threads(f)
    @test_call NT.enable_full_threading()

    # Dynamic dispatch through the registry's `Function` fields is expected; see above.
    @test_opt broken = true NT.with_thread_budget(f, 2)
end

@testset "JET: macro expansions" begin
    function budgeted_loop(out)
        @budgeted_threads for i in eachindex(out)
            out[i] = i
        end
        return out
    end
    function sequential_loop(out)
        @budgeted_threads threads = false for i in eachindex(out)
            out[i] = i
        end
        return out
    end

    @test_call budgeted_loop(zeros(Int, 4))
    @test_call sequential_loop(zeros(Int, 4))
    @test budgeted_loop(zeros(Int, 4)) == 1:4
    @test sequential_loop(zeros(Int, 4)) == 1:4
end
