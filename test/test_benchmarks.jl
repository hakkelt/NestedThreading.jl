# Local-only benchmarks: these compare wall-clock times and are therefore meaningful only on
# an otherwise idle machine. `test/runtests.jl` excludes the `:benchmark` tag by default;
# run them explicitly with
#
#     julia -t 8 --project=test test/runtests.jl :benchmark
#
# They need more than one thread to say anything; with `nthreads() == 1` they degrade to a
# correctness check.

@testmodule BenchHelpers begin
    using LinearAlgebra, Printf

    "Best-of-`reps` wall time of `f(args...)`, in seconds."
    function best_time(f, args...; reps = 5)
        return minimum(1:reps) do _
            t = time_ns()
            f(args...)
            return (time_ns() - t) / 1.0e9
        end
    end

    function report(label, baseline, budgeted)
        @printf(
            "  %-28s unbudgeted %7.4f s   budgeted %7.4f s   speedup %5.2fx\n",
            label, baseline, budgeted, baseline / budgeted
        )
        return baseline / budgeted
    end
end

@testitem "BLAS oversubscription is avoided" tags = [:benchmark] setup = [BenchHelpers] begin
    using NestedThreading, LinearAlgebra, Random

    nt = Threads.nthreads()
    Random.seed!(0)

    # One GEMM per outer worker, each large enough that BLAS would happily use every core.
    # Unbudgeted, the nt outer tasks each ask BLAS for nt threads => nt² runnable threads.
    n = 384
    As = [randn(n, n) for _ in 1:nt]
    Bs = [randn(n, n) for _ in 1:nt]
    Cs = [zeros(n, n) for _ in 1:nt]

    function unbudgeted!(Cs, As, Bs, reps)
        Threads.@threads for j in eachindex(Cs)
            for _ in 1:reps
                mul!(Cs[j], As[j], Bs[j])
            end
        end
        return Cs
    end

    function budgeted!(Cs, As, Bs, reps)
        @budgeted_threads for j in eachindex(Cs)
            for _ in 1:reps
                mul!(Cs[j], As[j], Bs[j])
            end
        end
        return Cs
    end

    BLAS.set_num_threads(nt)
    unbudgeted!(Cs, As, Bs, 1)                       # warm up both paths
    budgeted!(Cs, As, Bs, 1)

    # Correctness first: budgeting must not change the result.
    expected = [As[j] * Bs[j] for j in eachindex(Cs)]
    budgeted!(Cs, As, Bs, 1)
    @test all(j -> Cs[j] ≈ expected[j], eachindex(Cs))

    baseline = BenchHelpers.best_time(unbudgeted!, Cs, As, Bs, 4)
    budgeted = BenchHelpers.best_time(budgeted!, Cs, As, Bs, 4)
    speedup = BenchHelpers.report("nt×GEMM($n)", baseline, budgeted)

    @test BLAS.get_num_threads() == nt               # restored after the budgeted loop

    if nt > 1
        # Measured repeatedly at 1.4x-3.5x on an 8-thread machine; 1.1x is a deliberately
        # loose floor so the assertion survives a moderately busy machine while still
        # failing if budgeting stops taking effect at all.
        @test speedup > 1.1
    end
end

# Report-only: this one deliberately makes no performance assertion.
#
# Polyester does *not* oversubscribe the way BLAS does. `disable_polyester_threads` works by
# reserving PolyesterWeave's worker slots, so a nested `@batch` inside an outer parallel loop
# already finds no free workers and degrades to a serial run on its own — the cooperative
# behaviour BLAS lacks. Budgeting it therefore buys predictability, not throughput, and is
# measurably *slower* on this workload (~0.6x): Polyester's serial fallback path is less well
# optimized than its parallel one, so forcing it costs more than the nesting would have.
#
# The item is kept because that number is worth knowing and worth noticing if it changes; it
# prints the ratio and asserts only correctness and that the guard actually took effect.
@testitem "nested Polyester: budgeting cost (report only)" tags = [:benchmark] setup = [BenchHelpers] begin
    using NestedThreading, Random
    import Polyester

    nt = Threads.nthreads()
    Random.seed!(0)

    # An inner Polyester loop inside an outer parallel loop: unbudgeted, both levels try to
    # occupy the machine. The budgeted version disables the inner level for the duration.
    outer, inner = nt, 1 << 14
    data = [randn(inner) for _ in 1:outer]
    out = zeros(outer)

    function unbudgeted!(out, data, reps)
        Threads.@threads for j in eachindex(data)
            acc = 0.0
            for _ in 1:reps
                Polyester.@batch reduction = ((+, acc),) for i in eachindex(data[j])
                    acc += sin(data[j][i])
                end
            end
            out[j] = acc
        end
        return out
    end

    function budgeted!(out, data, reps)
        @budgeted_threads for j in eachindex(data)
            acc = 0.0
            for _ in 1:reps
                Polyester.@batch reduction = ((+, acc),) for i in eachindex(data[j])
                    acc += sin(data[j][i])
                end
            end
            out[j] = acc
        end
        return out
    end

    unbudgeted!(out, data, 1)
    budgeted!(out, data, 1)

    expected = [sum(sin, d) for d in data]
    budgeted!(out, data, 1)
    @test all(j -> out[j] ≈ expected[j], eachindex(out))

    baseline = BenchHelpers.best_time(unbudgeted!, out, data, 8)
    budgeted = BenchHelpers.best_time(budgeted!, out, data, 8)
    BenchHelpers.report("nt×(@batch over $inner)", baseline, budgeted)

    # The guard must actually have taken effect: inside a budgeted loop a nested `@batch`
    # runs on a single thread. This is the deterministic claim; the timing above is context.
    if nt > 1
        distinct = zeros(Int, nt)
        @budgeted_threads for j in 1:nt
            acc = zeros(Int, 64 * nt)
            Polyester.@batch for i in eachindex(acc)
                acc[i] = Threads.threadid()
            end
            distinct[j] = length(unique(acc))
        end
        @test all(==(1), distinct)
    end
end
