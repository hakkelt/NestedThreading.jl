# A note on what is and is not checked here.
#
# `CountedPool.get`/`set` and `GuardedPool.guard` are abstractly-typed `Function` fields.
# A registry that downstream packages extend at load time cannot be concretely typed, so
# calling through them is a runtime dispatch — see the long comment in `src/registry.jl`
# for why `FunctionWrapper` fields make this worse rather than better. It costs one dispatch
# per budget scope (once per `mul!`), never one per loop iteration, and it is confined to
# the `@noinline` helpers so it does not leak into callers' inlined code.
#
# Consequently `@test_opt` is applied to the parts that must stay statically resolved — the
# budget arithmetic and the macro expansions — while `@test_call` covers the full public
# surface. There is deliberately no `broken = true` marker for the registry dispatch: it is
# expected behaviour, documented above, not a defect awaiting a fix.

@testitem "JET: package" tags = [:jet] begin
    using NestedThreading, JET
    JET.test_package(NestedThreading; target_modules = (NestedThreading,))
end

@testitem "JET: budget arithmetic is statically resolved" tags = [:jet] begin
    using NestedThreading, JET
    const NT = NestedThreading

    # target_modules restricts JET to dispatches originating in our own code: on Julia
    # 1.10, Threads.threadpoolsize() itself takes an internal sprint-based fallback path
    # that JET flags as a runtime dispatch, even though nothing in NestedThreading forces
    # it. That is a stdlib implementation detail of the LTS release, not a regression we
    # can fix by choosing a different accessor.
    @test_opt target_modules = (NT,) NT.budget_for(1:8)
    @test_opt target_modules = (NT,) NT.budget_for(CartesianIndices((4, 4)))
    @test_opt target_modules = (NT,) NT.budget_for(Iterators.filter(isodd, 1:10))
    @test_opt target_modules = (NT,) NT.capacity()
    @test_call target_modules = (NT,) NT.budget_for(1:8)
end

@testitem "JET: scoped entry points" tags = [:jet] begin
    using NestedThreading, JET
    const NT = NestedThreading

    f() = 1 + 1
    @test_call NT.with_thread_budget(f, 2)
    @test_call NT.with_restricted_threads(f)
    @test_call NT.with_full_threads(f)
    @test_call NT.enable_full_threading()
    @test_call NT.register_counted_pool!(() -> 1, identity; name = :jet_probe)
end

@testitem "JET: macro expansions" tags = [:jet] begin
    using NestedThreading, JET
    import Polyester

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
    function batched_loop(out)
        @budgeted_batch for i in eachindex(out)
            out[i] = i
        end
        return out
    end

    @test_call budgeted_loop(zeros(Int, 4))
    @test_call sequential_loop(zeros(Int, 4))
    @test_call batched_loop(zeros(Int, 4))
    @test budgeted_loop(zeros(Int, 4)) == 1:4
    @test sequential_loop(zeros(Int, 4)) == 1:4
    @test batched_loop(zeros(Int, 4)) == 1:4
end
