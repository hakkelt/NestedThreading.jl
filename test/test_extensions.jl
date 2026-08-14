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
        for start in (true, false)
            NFFT._use_threads[] = start
            NT.with_restricted_threads() do
                @test NFFT._use_threads[] == false
            end
            @test NFFT._use_threads[] == start
        end

        # An explicit full-threads request turns NFFT on even if it was off, which is what
        # an operator with `threaded = true` needs.
        NFFT._use_threads[] = false
        NT.with_full_threads() do
            @test NFFT._use_threads[] == true
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
