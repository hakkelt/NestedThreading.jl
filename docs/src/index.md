# NestedThreading.jl

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

Support for a library is loaded automatically when that library is loaded; see
[Supported libraries](@ref).

## Quick start

Replace `Threads.@threads` with [`@budgeted_threads`](@ref):

```jldoctest
julia> using NestedThreading, LinearAlgebra

julia> A = rand(64, 64); X = rand(64, 32); Y = similar(X);

julia> @budgeted_threads for i in axes(X, 2)
           mul!(view(Y, :, i), A, view(X, :, i))
       end

julia> Y ≈ A * X
true
```

The loop has 32 iterations. On 8 threads each worker gets `max(1, 8 ÷ 32) = 1` thread, so
BLAS runs single-threaded inside the loop. A loop of 2 iterations would give each worker 4
BLAS threads. The previous thread counts come back when the loop ends.

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
[Design](@ref) has the full table.

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

## Where to go next

- [How-to guides](@ref "Budget a parallel loop"): budget loops, restrict other calls, register a library.
- [Design](@ref): why limits work the way they do.
- [Composition rules](@ref) and the [API](@ref "API reference"): exact behavior.

!!! note "Disclaimer"
    This package was written mostly by Claude (Anthropic's coding assistant), under human
    direction and review. Treat it as you would any other third-party dependency and read
    the code before relying on it.
