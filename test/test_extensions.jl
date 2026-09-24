@testitem "extensions load and register their pools" tags = [:extensions] begin
    using NestedThreading
    using FFTW
    import NFFT, Polyester
    const NT = NestedThreading

    ext(name) = Base.get_extension(NestedThreading, name)
    counted = [p.name for p in NT.COUNTED_POOLS]
    guarded = [p.name for p in NT.GUARDED_POOLS]

    @test ext(:NestedThreadingFFTWExt) !== nothing
    @test :fftw in counted
    @test ext(:NestedThreadingPolyesterExt) !== nothing
    @test :polyester in guarded
    @test ext(:NestedThreadingNFFTExt) !== nothing
    @test :nfft in counted
end

# The MKL extension is tested separately (not here): MKL_jll ships no artifact for macOS or
# any non-x86_64 platform, so a `using MKL` this test environment would fail to precompile
# on those CI runners. See the "MKL pool" step in the Tests workflow, which runs only on the
# platforms MKL_jll actually supports.

@testitem "FFTW pool" tags = [:extensions] begin
    using NestedThreading, FFTW
    const NT = NestedThreading

    original = FFTW.get_num_threads()
    NT.with_thread_budget(2) do
        @test FFTW.get_num_threads() == 2
    end
    @test FFTW.get_num_threads() == original
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
            NT.with_restricted_threads() do
                @test NFFT._use_threads[] == false
            end
            @test NFFT._use_threads[] == start
            NT.with_full_threads() do
                nothing
            end
            @test NFFT._use_threads[] == start
        end

        # An explicit full-threads request turns NFFT on even if it was off, which is what
        # an operator with `threaded = true` needs. Not in a single-threaded session, where
        # NFFT's threaded path has no workers to use.
        NFFT._use_threads[] = false
        NT.with_full_threads() do
            @test NFFT._use_threads[] == (NT.capacity() > 1)
        end
        @test NFFT._use_threads[] == false

        # A partial budget is still "restricted" for a pool with no partial control.
        if NT.capacity() > 2
            NFFT._use_threads[] = true
            NT.with_thread_budget(2) do
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
        return NT.with_thread_budget(states, 2)
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
