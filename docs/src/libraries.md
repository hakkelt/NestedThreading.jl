# Supported libraries

Each library is registered when its package is loaded.

| Library | Name | Controls | What is called |
| --- | --- | --- | --- |
| BLAS | `:blas` | `get`/`set`, `park` | `BLAS.get_num_threads`/`set_num_threads`; `park` shuts OpenBLAS's workers down. Always registered. |
| MKL | `:mkl` | `get`/`set` | `MKL.get_num_threads`/`set_num_threads` |
| FFTW | `:fftw` | `get`/`set` | `FFTW.get_num_threads`/`set_num_threads` |
| NFFT | `:nfft` | `get`/`set` (on/off) | `NFFT._use_threads[]` |
| Polyester, LoopVectorization, Octavian | `:polyester` | `guard`, `quiesce` | reserves every PolyesterWeave worker, as `disable_polyester_threads` does |
| Strided | `:strided` | `get`/`set` | `Strided.get_num_threads`/`set_num_threads`, capped at `Threads.nthreads()` |

MKL has its own entry besides `:blas` because `BLAS.set_num_threads` does not reliably
change MKL's own count
([MKL.jl#174](https://github.com/JuliaLinearAlgebra/MKL.jl/issues/174)).
