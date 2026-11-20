# NestedThreading.jl

[![Tests](https://github.com/hakkelt/NestedThreading.jl/actions/workflows/tests.yml/badge.svg)](https://github.com/hakkelt/NestedThreading.jl/actions/workflows/tests.yml)
[![Docs (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://hakkelt.github.io/NestedThreading.jl/stable/)
[![Docs (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://hakkelt.github.io/NestedThreading.jl/dev/)
[![codecov](https://codecov.io/gh/hakkelt/NestedThreading.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/hakkelt/NestedThreading.jl)
[![version](https://juliahub.com/docs/General/NestedThreading/stable/version.svg)](https://juliahub.com/ui/Packages/General/NestedThreading)
[![Aqua QA](https://juliatesting.github.io/Aqua.jl/dev/assets/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

BLAS, FFTW, Polyester, NFFT and similar libraries each choose their own thread count. Called
from inside a `Threads.@threads` loop that already uses every thread, each worker starts its
own inner threads, and the machine runs far more threads than it has cores. NestedThreading
limits the inner libraries for the duration of the outer loop, the way Python's
[`threadpoolctl`](https://github.com/joblib/threadpoolctl) does.

## Installation

```julia
using Pkg
Pkg.add("NestedThreading")
```

## Example

```julia
using NestedThreading, LinearAlgebra

A = rand(64, 64); X = rand(64, 32); Y = similar(X)

# 32 iterations on 8 threads: BLAS gets max(1, 8 ÷ 32) = 1 thread per worker.
@budgeted_threads for i in axes(X, 2)
    mul!(view(Y, :, i), A, view(X, :, i))
end
```

## Why it matters

Two loops that each run a threaded library inside `Threads.@threads`, timed with
BenchmarkTools on a 48-core machine. The comments give the time at `julia -t 48` and at
`julia -t 8`.

```julia
using NestedThreading, LinearAlgebra, BenchmarkTools
import Polyester

nt = Threads.nthreads()

# BLAS inside a threaded loop: four 384×384 matrix products per thread.
As, Bs, Cs = ([randn(384, 384) for _ in 1:nt] for _ in 1:3)
BLAS.set_num_threads(nt)

function gemm_plain!(Cs, As, Bs)
    Threads.@threads for j in eachindex(Cs)
        for _ in 1:4
            mul!(Cs[j], As[j], Bs[j])
        end
    end
end

function gemm_budgeted!(Cs, As, Bs)
    @budgeted_threads for j in eachindex(Cs)
        for _ in 1:4
            mul!(Cs[j], As[j], Bs[j])
        end
    end
end

@btime gemm_plain!($Cs, $As, $Bs)        # -t 48: 86 s       -t 8: 28 ms
@btime gemm_budgeted!($Cs, $As, $Bs)     # -t 48: 0.10 s     -t 8: 19 ms

# Polyester inside a threaded loop: a sum over 16384 elements, eight times per thread.
data = [randn(2^14) for _ in 1:nt]
out = zeros(nt)

function batch_plain!(out, data)
    Threads.@threads for j in eachindex(data)
        acc = 0.0
        for _ in 1:8
            Polyester.@batch reduction = ((+, acc),) for i in eachindex(data[j])
                acc += sin(data[j][i])
            end
        end
        out[j] = acc
    end
end

function batch_budgeted!(out, data)
    @budgeted_threads for j in eachindex(data)
        acc = 0.0
        for _ in 1:8
            Polyester.@batch reduction = ((+, acc),) for i in eachindex(data[j])
                acc += sin(data[j][i])
            end
        end
        out[j] = acc
    end
end

@btime batch_plain!($out, $data)         # -t 48: 1.1 s      -t 8: 2.6 ms
@btime batch_budgeted!($out, $data)      # -t 48: 13.5 ms    -t 8: 4.1 ms
```

When Julia uses every core, the unbudgeted loops are 846x and 82x slower. With cores to
spare, nesting costs little, and budgeting the Polyester loop makes it slower.
The [design page](https://hakkelt.github.io/NestedThreading.jl/stable/design/) has the full table.

Polyester's workers keep spinning on Julia's threads for a while after a `@batch` loop ends,
and a `Threads.@threads` loop started in that window waits for them. The budgeted macros
release them first:

```julia
xs = rand(2^12)

function side_by_side_plain!(xs)
    Polyester.@batch for i in eachindex(xs)
        xs[i] = sin(xs[i])
    end
    Threads.@threads for i in eachindex(xs)
        xs[i] = cos(xs[i])
    end
end

function side_by_side_budgeted!(xs)
    @budgeted_batch for i in eachindex(xs)
        xs[i] = sin(xs[i])
    end
    @budgeted_threads for i in eachindex(xs)
        xs[i] = cos(xs[i])
    end
end

@btime side_by_side_plain!($xs)          # -t 8: 70–130 µs
@btime side_by_side_budgeted!($xs)       # -t 8: 35 µs
```

## API

| Function / macro | Purpose |
| --- | --- |
| `@budgeted_threads [schedule] [threads=…] for …` | `Threads.@threads` with a thread budget for inner libraries and an on/off switch |
| `@budgeted_batch [options…] [threads=…] for …` | the same for Polyester's `@batch` |
| `@budgeted <loop-macro> for …` | budget a loop written with any other macro |
| `with_threads_disabled(f)` | one thread for every library while `f()` runs |
| `with_threads_enabled(f)` | every thread, within any enclosing limit |
| `with_thread_limit(f, n)` | at most `n` threads; nested scopes cannot raise it |
| `with_thread_default(f, n)` | `n` threads unless a grant inside asks for more |
| `with_thread_grant(f, n)` | `n` threads despite a default, within any limit |
| `thread_pool_info([name])` | registered libraries and their thread counts |

Public, not exported: `register_pool!`, `quiesce_foreign_pools`, `reset_thread_counts!`,
`thread_limit`, `capacity`. Every scope takes `only` and `exclude` to choose which libraries
it restricts.

## Supported libraries

BLAS (always), MKL, FFTW, NFFT, Polyester (with LoopVectorization and Octavian) and Strided,
each registered when its package is loaded. Other libraries can be added with
`NestedThreading.register_pool!`.

See the [documentation](https://hakkelt.github.io/NestedThreading.jl/stable/) for guides,
the composition rules and benchmarks.

## Tests

```sh
julia --project=test test/runtests.jl              # everything except :benchmark
julia --project=test test/runtests.jl :macros      # one tag
julia --project=test test/runtests.jl :benchmark   # timing comparisons
```

Tags: `:registry`, `:macros`, `:extensions`, `:jet`, `:aqua`, `:benchmark`. Also run with
`julia -t 1`: `capacity() == 1` is a separate code path.

## License

MIT — see [LICENSE.md](LICENSE.md).

> **Disclaimer:** this package was written mostly by Claude (Anthropic's coding assistant),
> under human direction and review. Treat it as you would any other third-party dependency
> and read the code before relying on it.
