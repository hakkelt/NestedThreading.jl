@testitem "extensions load and register their pools" tags = [:extensions] begin
    using NestedThreading
    using FFTW
    import NFFT, Polyester, Strided
    const NT = NestedThreading

    ext(name) = Base.get_extension(NestedThreading, name)
    counted = [p.name for p in NT.COUNTED_POOLS]
    guarded = [p.name for p in NT.GUARDED_POOLS]

    @test ext(:NestedThreadingFFTWExt) !== nothing
    @test :fftw in counted
    @test ext(:NestedThreadingPolyesterWeaveExt) !== nothing
    @test :polyester in guarded
    @test ext(:NestedThreadingNFFTExt) !== nothing
    @test :nfft in counted
    @test ext(:NestedThreadingStridedExt) !== nothing
    @test :strided in counted
end

# The MKL extension is tested separately (not here): MKL_jll ships no artifact for macOS or
# any non-x86_64 platform, so a `using MKL` this test environment would fail to precompile
# on those CI runners. See the "MKL pool" step in the Tests workflow, which runs only on the
# platforms MKL_jll actually supports.

@testitem "FFTW pool" tags = [:extensions] begin
    using NestedThreading, FFTW
    const NT = NestedThreading

    original = FFTW.get_num_threads()
    NT.with_thread_limit(2) do
        @test FFTW.get_num_threads() == 2
    end
    @test FFTW.get_num_threads() == original
end

@testitem "Strided pool" tags = [:extensions] begin
    using NestedThreading
    import Strided
    const NT = NestedThreading

    original = Strided.get_num_threads()
    NT.with_threads_disabled() do
        @test Strided.get_num_threads() == 1
    end
    @test Strided.get_num_threads() == original
    NT.with_thread_limit(2) do
        @test Strided.get_num_threads() == min(2, Threads.nthreads())
    end
    @test Strided.get_num_threads() == original
    # A budget above Julia's thread count is clamped, without Strided's warning.
    @test_logs NT.with_thread_limit(Threads.nthreads() + 1) do
        @test Strided.get_num_threads() == Threads.nthreads()
    end
    @test Strided.get_num_threads() == original
end

@testitem "NFFT pool is all-or-nothing and round-trips" tags = [:extensions] begin
    using NestedThreading
    import NFFT
    const NT = NestedThreading

    # NFFT is registered as a *counted* pool rather than a guard: `_use_threads[]` has an
    # imperative setter, so routing it through the refcounted registry keeps it safe under
    # concurrency instead of having every scope save and restore the global itself.
    original = NFFT._use_threads[]
    try
        # The setting must survive a budget scope whatever it started as, and whatever the
        # thread count — including `capacity() == 1`, where "restricted" and "full" are the
        # same budget and a naive `n >= capacity()` mapping flips `false` to `true` and
        # never restores it. See the extension for why the threshold is clamped to 2.
        for start in (true, false)
            NFFT._use_threads[] = start
            NT.with_threads_disabled() do
                @test NFFT._use_threads[] == false
            end
            @test NFFT._use_threads[] == start
            NT.with_threads_enabled() do
                nothing
            end
            @test NFFT._use_threads[] == start
        end

        # An explicit full-threads request turns NFFT on even if it was off, which is what
        # an operator with `threaded = true` needs. Not in a single-threaded session, where
        # NFFT's threaded path has no workers to use.
        NFFT._use_threads[] = false
        NT.with_threads_enabled() do
            @test NFFT._use_threads[] == (NT.capacity() > 1)
        end
        @test NFFT._use_threads[] == false

        # A partial budget is still "restricted" for a pool with no partial control.
        if NT.capacity() > 2
            NFFT._use_threads[] = true
            NT.with_thread_limit(2) do
                @test NFFT._use_threads[] == false
            end
            @test NFFT._use_threads[] == true
        end
    finally
        NFFT._use_threads[] = original
    end
end

@testitem "quiescing Polyester never drops a concurrent @batch chunk" tags = [:extensions] begin
    using NestedThreading, Polyester
    const NT = NestedThreading

    # Every other task runs a `@batch` while its neighbours quiesce. A quiesce that parks a
    # worker another task has just launched a chunk on makes that chunk vanish: the worker runs
    # the park request instead, the launching task sees it finish, and part of its output is
    # never written.
    if NT.capacity() > 2
        n = 1 << 14
        dropped = Threads.Atomic{Int}(0)
        for _ in 1:200
            Threads.@threads for k in 1:(2 * Threads.nthreads())
                if isodd(k)
                    y = fill(NaN, n)
                    Polyester.@batch for i in 1:n
                        y[i] = i
                    end
                    any(isnan, y) && Threads.atomic_add!(dropped, 1)
                else
                    NT.quiesce_foreign_pools()
                end
            end
        end
        @test dropped[] == 0
    end
end

@testitem "a restricted scope parks the Polyester workers it reserves" tags = [:extensions] begin
    using NestedThreading, Polyester
    const NT = NestedThreading
    const TU = Polyester.ThreadingUtilities

    states() = [TU._atomic_state(TU.taskpointer(tid)) for tid in eachindex(TU.TASKS)]
    # In a function, so that the scope opens while the workers are still spinning after the
    # loop rather than after they have given up and parked by themselves.
    function states_inside_scope(y)
        Polyester.@batch for i in eachindex(y)
            y[i] = i
        end
        return NT.with_thread_limit(states, 2)
    end

    if NT.capacity() > 2
        y = zeros(1 << 12)
        states_inside_scope(y)
        @test all(==(TU.WAIT), states_inside_scope(y))
        # Released, not left reserved: a later loop still spreads over them.
        tids = zeros(Int, Threads.nthreads())
        Polyester.@batch per = thread for i in eachindex(tids)
            tids[i] = Threads.threadid()
        end
        @test length(unique(tids)) > 1
    end
end

@testitem "the PolyesterWeave guard covers LoopVectorization's @tturbo" tags = [:extensions] begin
    using NestedThreading, LoopVectorization
    const NT = NestedThreading
    const PW = LoopVectorization.PolyesterWeave

    # `@tturbo` takes its workers from `PolyesterWeave.request_threads`, the same slots the
    # guard reserves, so inside a restricted scope it finds none free and runs serially.
    function free_workers()
        masks = last(PW.request_threads(Threads.nthreads()))
        n = sum(count_ones, masks)
        foreach(PW.free_threads!, masks)
        return n
    end
    if NT.capacity() > 1
        @test free_workers() > 0
        @test NT.with_thread_limit(free_workers, 1) == 0
        @test NT.with_thread_limit(free_workers, NT.capacity() - 1) == 0
        @test free_workers() > 0
    end

    function double!(y, x)
        @tturbo for i in eachindex(y)
            y[i] = 2 * x[i]
        end
        return y
    end
    xs = [rand(1 << 14) for _ in 1:Threads.nthreads()]
    ys = map(similar, xs)
    @budgeted_threads for j in eachindex(xs)
        double!(ys[j], xs[j])
    end
    @test all(ys[j] == 2 .* xs[j] for j in eachindex(xs))
end

@testitem "the PolyesterWeave extension loads without Polyester" tags = [:extensions] begin
    # LoopVectorization pulls in PolyesterWeave and ThreadingUtilities but not Polyester; the
    # extension must load from those indirect dependencies alone.
    code = """
    using NestedThreading, LoopVectorization
    print(Base.get_extension(NestedThreading, :NestedThreadingPolyesterWeaveExt) !== nothing,
          " ", !any(id -> id.name == "Polyester", keys(Base.loaded_modules)),
          " ", :polyester in [p.name for p in NestedThreading.GUARDED_POOLS])
    """
    cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(Base.active_project()) -e $code`
    @test read(cmd, String) == "true true true"
end
