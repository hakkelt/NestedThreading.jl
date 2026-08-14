# Local-only benchmarks: these compare wall-clock times and are therefore meaningful only on
# an otherwise idle machine. `test/runtests.jl` excludes the `:benchmark` tag by default;
# run them explicitly with
#
#     julia -t 8 --project=test test/runtests.jl :benchmark
#
# They need more than one thread to say anything; with `nthreads() == 1` they degrade to a
# correctness check.
#
# Timings come from BenchmarkTools (`@belapsed`, i.e. the minimum over the samples it tunes
# for itself). The workloads mutate preallocated buffers, so every benchmark interpolates its
# arguments with `$` and uses `evals = 1` — re-running the same in-place kernel many times
# within one eval would measure warm caches rather than the work.

@testmodule BenchHelpers begin
    using BenchmarkTools, Printf, Statistics

    "Seconds of measurement budget per individual benchmark."
    const BUDGET = 0.5

    "Number of alternating baseline/variant rounds per comparison."
    const ROUNDS = 5

    """
        compare(call, baseline, variant; rounds = ROUNDS) -> (ratio, t_base, t_var)

    Median of `rounds` *paired* measurements: baseline and variant are timed back-to-back
    within each round and the ratio is taken per round, so slow drift between rounds cancels
    out.

    This matters more than it sounds. On a dual-socket machine the same threaded loop is
    bimodal — here roughly 1.8 ms or 3.0 ms depending on how the worker threads land across
    sockets — and that bimodality is far larger than the effects being measured. Comparing
    two benchmarks taken minutes apart straddles it and produces ratios like 0.59x for two
    loops that are behaviourally identical (verified: same allocation count, ratio 1.00x
    when paired).

    `call(f)` must invoke `f` on the shared, preallocated workload.
    """
    function compare(call, baseline, variant; rounds = ROUNDS)
        ratios, bases, vars = Float64[], Float64[], Float64[]
        for _ in 1:rounds
            tb = @belapsed $call($baseline) evals = 1 seconds = BUDGET
            tv = @belapsed $call($variant) evals = 1 seconds = BUDGET
            push!(ratios, tv / tb)
            push!(bases, tb)
            push!(vars, tv)
        end
        return median(ratios), median(bases), median(vars)
    end

    """
        warmup!(call, fs; rounds = 3)

    Run every function in `fs` a few times before measuring anything. BenchmarkTools warms up
    each benchmark on its own, but the first benchmark in a process additionally pays for
    thread-pool spin-up, which would otherwise be charged to whichever variant is measured
    first.
    """
    function warmup!(call, fs; rounds = 3)
        for _ in 1:rounds, f in fs
            call(f)
        end
        return nothing
    end

    function report(label, t_base, t_var, ratio)
        @printf(
            "  %-28s baseline %8.5f s   variant %8.5f s   variant/baseline %5.2fx\n",
            label, t_base, t_var, ratio
        )
        return ratio
    end
end

@testitem "BLAS oversubscription is avoided" tags = [:benchmark] setup = [BenchHelpers] begin
    using NestedThreading, BenchmarkTools, LinearAlgebra, Random

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

    # Correctness first: budgeting must not change the result.
    expected = [As[j] * Bs[j] for j in eachindex(Cs)]
    budgeted!(Cs, As, Bs, 1)
    @test all(j -> Cs[j] ≈ expected[j], eachindex(Cs))

    # Warm up *both* paths before any measurement. BenchmarkTools warms up each benchmark
    # individually, but the first one in a process also pays for thread-pool spin-up, which
    # otherwise lands entirely on whichever variant happens to be measured first.
    call(f) = f(Cs, As, Bs, 4)
    BenchHelpers.warmup!(call, (unbudgeted!, budgeted!))

    ratio, t_unb, t_bud = BenchHelpers.compare(call, unbudgeted!, budgeted!)
    BenchHelpers.report("nt×GEMM($n)", t_unb, t_bud, ratio)
    speedup = 1 / ratio

    @test BLAS.get_num_threads() == nt               # restored after the budgeted loop

    if nt > 1
        # Measured repeatedly at 1.4x-3.5x on an idle 8-thread machine; 1.1x is a
        # deliberately loose floor so the assertion survives a moderately busy machine while
        # still failing if budgeting stops taking effect at all.
        @test speedup > 1.1
    end
end

# Budgeting a loop whose *inner* parallelism is Polyester rather than BLAS measures at ~0.6x
# on this workload — a real loss. This item exists to attribute that loss, because the
# interesting question is whether it comes from this package's machinery or from serializing
# the inner `@batch`.
#
# It is the latter. The three controls below all land on 1.00x against a plain
# `Threads.@threads` loop, and they are asserted, not just printed:
#
#   * a do-nothing closure wrapper            -> the `do` block costs nothing
#   * a budget scope with :polyester excluded -> the lock/refcount/counted pools cost nothing
#   * the Polyester guard around a loop with no `@batch` in it -> the guard call costs nothing
#
# So the whole difference is the effect of serializing the inner `@batch`. Note that
# Polyester's serial path is *not* intrinsically slow: measured against a hand-written `for`
# loop in isolation it is within 3%. The mechanism behind the loss when many threads run
# serialized `@batch`es concurrently has not been isolated, and nothing here claims to know
# it — hence no performance assertion on the headline ratio, only on the controls.
@testitem "nested Polyester: budgeting cost, attributed" tags = [:benchmark] setup = [BenchHelpers] begin
    using NestedThreading, BenchmarkTools, Random
    import Polyester

    nt = Threads.nthreads()
    Random.seed!(0)

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

    expected = [sum(sin, d) for d in data]
    budgeted!(out, data, 1)
    @test all(j -> out[j] ≈ expected[j], eachindex(out))

    call(f) = f(out, data, 8)
    BenchHelpers.warmup!(call, (unbudgeted!, budgeted!))

    ratio, t_unb, t_bud = BenchHelpers.compare(call, unbudgeted!, budgeted!)
    BenchHelpers.report("nt×(@batch over $inner)", t_unb, t_bud, ratio)

    # --- attribution controls -----------------------------------------------------------
    #
    # Same outer loop, but with a Polyester-free inner body, so the only thing that varies is
    # how the loop is wrapped. If any of these drifts away from 1.00x, this package has
    # started costing something, and that is a defect worth failing on.
    serial_kern(d, reps) = begin
        acc = 0.0
        for _ in 1:reps, i in eachindex(d)
            acc += sin(d[i])
        end
        acc
    end

    plain!(out, data, reps) =
        Threads.@threads for j in eachindex(data)
            out[j] = serial_kern(data[j], reps)
        end
    noop_wrapped!(out, data, reps) =
        ((f) -> f())() do
            Threads.@threads for j in eachindex(data)
                out[j] = serial_kern(data[j], reps)
            end
        end
    budget_no_guard!(out, data, reps) =
        NestedThreading.with_thread_budget(1; exclude = (:polyester,)) do
            Threads.@threads for j in eachindex(data)
                out[j] = serial_kern(data[j], reps)
            end
        end
    guard_only!(out, data, reps) =
        Polyester.disable_polyester_threads() do
            Threads.@threads for j in eachindex(data)
                out[j] = serial_kern(data[j], reps)
            end
        end

    BenchHelpers.warmup!(call, (plain!, noop_wrapped!, budget_no_guard!, guard_only!))

    for (label, variant) in (
            ("closure wrapper only", noop_wrapped!),
            ("budget, :polyester excluded", budget_no_guard!),
            ("guard, no @batch inside", guard_only!),
        )
        r, tb, tv = BenchHelpers.compare(call, plain!, variant)
        BenchHelpers.report(label, tb, tv, r)
        # Paired and median-reduced, so this bound is about real overhead rather than
        # machine drift. The claim being defended is "no measurable overhead".
        nt > 1 && @test r < 1.15
    end

    # The guard must also actually take effect: inside a budgeted loop a nested `@batch` runs
    # on a single thread.
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
