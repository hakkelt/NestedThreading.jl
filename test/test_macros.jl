@testitem "@budgeted computes the budget from the trip count" tags = [:macros] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    n = NT.capacity()

    seen = zeros(Int, 2n)
    @budgeted Threads.@threads for j in 1:(2n)
        seen[j] = Probe.VALUE[]
    end
    @test all(==(1), seen)                    # the loop already saturates the machine
    @test Probe.VALUE[] == 16

    seen2 = zeros(Int, 1)
    @budgeted Threads.@threads for j in 1:1
        seen2[j] = Probe.VALUE[]
    end
    @test seen2[1] == n                       # single item: the body may use everything
end

@testitem "the range expression is evaluated exactly once" tags = [:macros] setup = [Probe] begin
    using NestedThreading
    Probe.reset!()

    evals = Ref(0)
    mkrange() = (evals[] += 1; 1:4)

    out = zeros(Int, 4)
    @budgeted_threads for j in mkrange()
        out[j] = j
    end
    @test evals[] == 1
    @test out == 1:4

    evals[] = 0
    out2 = zeros(Int, 4)
    @budgeted_threads threads = false for j in mkrange()
        out2[j] = j
    end
    @test evals[] == 1                        # also once on the sequential branch
    @test out2 == 1:4
end

@testitem "@budgeted_threads switch" tags = [:macros] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    n = NT.capacity()

    budgets = zeros(Int, 2)
    @budgeted_threads for j in 1:2
        budgets[j] = Probe.VALUE[]
    end
    @test all(==(max(1, n ÷ 2)), budgets)

    # threads = false runs sequentially *and* restricts inner libraries to one thread.
    budgets2 = zeros(Int, 2)
    tids = zeros(Int, 2)
    @budgeted_threads threads = false for j in 1:2
        budgets2[j] = Probe.VALUE[]
        tids[j] = Threads.threadid()
    end
    @test all(==(1), budgets2)
    @test length(unique(tids)) == 1
    @test Probe.VALUE[] == 16

    # The switch is a runtime expression, not a literal.
    flag = false
    tids2 = zeros(Int, 4)
    @budgeted_threads threads = flag for j in 1:4
        tids2[j] = Threads.threadid()
    end
    @test length(unique(tids2)) == 1
end

@testitem "@budgeted accepts macro chains" tags = [:macros] setup = [Probe] begin
    using NestedThreading
    const NT = NestedThreading
    Probe.reset!()

    out = zeros(Int, 4)
    @budgeted Threads.@threads :static for j in 1:4
        @inbounds out[j] = Probe.VALUE[]
    end
    @test all(==(max(1, NT.capacity() ÷ 4)), out)

    # A runtime (non-literal) range works, since the budget is computed at runtime.
    len = 2
    out2 = zeros(Int, len)
    @budgeted Threads.@threads for j in 1:len
        out2[j] = Probe.VALUE[]
    end
    @test all(==(max(1, NT.capacity() ÷ 2)), out2)
end

@testitem "short aliases expand identically" tags = [:macros] setup = [Probe] begin
    using NestedThreading
    Probe.reset!()

    a = zeros(Int, 2)
    b = zeros(Int, 2)
    @budgeted_threads for j in 1:2
        a[j] = Probe.VALUE[]
    end
    @ntt for j in 1:2
        b[j] = Probe.VALUE[]
    end
    @test a == b
end

@testitem "malformed input is rejected at expansion" tags = [:macros] begin
    using NestedThreading

    expand(str) = macroexpand(@__MODULE__, Meta.parse(str))

    @test_throws Exception expand("@budgeted Threads.@threads 1 + 1")
    @test_throws Exception expand("@budgeted_threads while true; break; end")
    # Multiple iteration specs have an ambiguous trip count.
    @test_throws Exception expand("@budgeted_threads for i in 1:2, j in 1:2; nothing; end")
    # `@budgeted` needs a loop *construct*; a bare `for` would silently be sequential.
    @test_throws Exception expand("@budgeted for i in 1:2; nothing; end")
    # An unrecognised keyword argument is an error, not silently ignored.
    @test_throws Exception expand("@budgeted_threads thread = true for i in 1:2; nothing; end")

    # ...and the well-formed shapes do expand.
    @test expand("@budgeted Threads.@threads for i in 1:2; nothing; end") isa Expr
    @test expand("@budgeted_threads threads = f() for i in 1:2; nothing; end") isa Expr
end

@testitem "@budgeted_batch does not disable its own loop" tags = [:macros] setup = [Probe] begin
    using NestedThreading
    import Polyester
    const NT = NestedThreading
    Probe.reset!()

    n = NT.capacity()
    tids = zeros(Int, 64 * n)
    @budgeted_batch for i in eachindex(tids)
        tids[i] = Threads.threadid()
    end
    @test Probe.VALUE[] == 16
    if n > 1
        @test length(unique(tids)) > 1        # the batch loop really ran in parallel
    end

    # A nested Polyester loop is switched off inside any restricted scope, regardless of how
    # much headroom the budget nominally leaves. Limiting it proportionally instead was
    # measured 320x worse in the saturated regime; see the Polyester extension.
    nested_threads(trip) = begin
        distinct = zeros(Int, trip)
        @budgeted_threads for j in 1:trip
            acc = zeros(Int, 64 * n)
            Polyester.@batch for k in eachindex(acc)
                acc[k] = Threads.threadid()
            end
            distinct[j] = length(unique(acc))
        end
        distinct
    end

    # Outer loop saturates the machine (budget 1).
    @test all(==(1), nested_threads(n))
    # Outer loop uses 2 of n threads (budget n ÷ 2) — still off, deliberately.
    n >= 4 && @test all(==(1), nested_threads(2))
end

@testitem "@budgeted_batch switch and chains" tags = [:macros] setup = [Probe] begin
    using NestedThreading
    import Polyester
    Probe.reset!()

    out = zeros(Int, 8)
    @budgeted_batch threads = false for i in 1:8
        out[i] = Probe.VALUE[]
    end
    @test all(==(1), out)

    acc = zeros(Int, 8)
    @budgeted @inbounds Polyester.@batch for i in 1:8
        acc[i] = i
    end
    @test acc == 1:8
end
