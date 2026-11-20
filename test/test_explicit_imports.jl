@testitem "ExplicitImports" tags = [:aqua] begin
    using NestedThreading, ExplicitImports
    # Load every weak dependency in the test environment so the checks also cover the
    # extensions.
    using FFTW
    import NFFT, Polyester, Strided

    # Non-public names the package reaches on purpose; none has a public equivalent.
    ignore = (
        # Base iterator traits: the budget dispatches on whether a loop's length is known.
        :HasLength, :HasShape,
        # BLAS library discovery, needed to call OpenBLAS's own thread shutdown.
        :get_config,
        # FFTW's and Strided's thread-count getters and setters.
        :get_num_threads, :set_num_threads,
        # NFFT's threading switch.
        :_use_threads,
        # ThreadingUtilities/ManualMemory internals used to park PolyesterWeave workers.
        :_sleep, :taskpointer, :_atomic_cas_cmp!, :SPIN, :TASK, :wait, :store!,
    )
    # Before Julia 1.11 there is no `public`, so these two checks would flag every
    # public-but-unexported name.
    test_explicit_imports(
        NestedThreading;
        ignore,
        all_explicit_imports_are_public = VERSION >= v"1.11",
        all_qualified_accesses_are_public = VERSION >= v"1.11",
    )
end
