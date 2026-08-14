@testset "extension loading" begin
    ext(name) = Base.get_extension(NestedThreading, Symbol(name))
    names = [p.name for p in NT.COUNTED_POOLS]
    gnames = [p.name for p in NT.GUARDED_POOLS]

    if HAS_FFTW
        @test ext(:NestedThreadingFFTWExt) !== nothing
        @test :fftw in names
    end
    if HAS_POLYESTER
        @test ext(:NestedThreadingPolyesterExt) !== nothing
        @test :polyester in gnames
    end
    if HAS_NFFT
        @test ext(:NestedThreadingNFFTExt) !== nothing
        @test :nfft in names
    end
end

if HAS_FFTW
    @testset "FFTW pool" begin
        original = FFTW.get_num_threads()
        NT.with_thread_budget(2) do
            @test FFTW.get_num_threads() == 2
        end
        @test FFTW.get_num_threads() == original
    end
end

if HAS_NFFT
    @testset "NFFT pool is all-or-nothing and round-trips" begin
        original = NFFT._use_threads[]
        try
            for start in (true, false)
                NFFT._use_threads[] = start
                NT.with_restricted_threads() do
                    @test NFFT._use_threads[] == false
                end
                @test NFFT._use_threads[] == start
            end

            # An explicit full-threads request turns NFFT on even if it was off, which is
            # what an operator with `threaded = true` needs.
            NFFT._use_threads[] = false
            NT.with_full_threads() do
                @test NFFT._use_threads[] == (NT.capacity() >= 1)
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
end
