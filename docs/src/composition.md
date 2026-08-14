# Composition rules

Everything in this package reduces to [`NestedThreading.with_thread_budget`](@ref). The rules
below are what make it safe to use from library code that does not know what its caller is
doing.

## Budgets are refcounted, process-wide

The registry keeps a multiset of every budget scope currently active anywhere in the process.

* On the transition from *no scope* to *one scope*, the current thread counts are
  snapshotted.
* While any scope is open, the applied count is `minimum` over all active budgets.
* On the transition back to *no scope*, the snapshot is restored.

## Nesting only ever narrows

```julia
with_thread_budget(1) do
    with_thread_budget(8) do
        BLAS.get_num_threads()   # == 1
    end
end
```

An inner scope asking for more threads cannot widen a restriction an outer caller is relying
on, because the applied value is the minimum. This matters for operator algebra: a batched
operator whose body contains another batched operator, or an NFFT operator constructed with
`threaded = true` that ends up being called from inside a saturated loop, both compose
correctly without either site knowing about the other.

## Concurrent scopes cannot corrupt each other

This is the failure mode the refcounting exists to prevent. With per-scope save/restore — even
with a lock around each half — two tasks interleave:

| | task A | task B | BLAS |
| --- | --- | --- | --- |
| 1 | saves 8, applies 1 | | 1 |
| 2 | | saves **1**, applies 1 | 1 |
| 3 | restores 8 | | 8 |
| 4 | | restores **1** | 1 |

and the process is left permanently throttled to 1 with no scope active. A mutex does not fix
this: it is the *pairing* of save and restore that interleaves, not the individual mutations.
Snapshotting once and restoring once removes the failure entirely.

## The cost: conservatism across unrelated tasks

BLAS and FFTW thread counts are process-global; the C libraries have no per-task scoping and
nothing in Julia can give them any. Taking the minimum over all active scopes therefore means:

!!! warning "While any task is restricted, unrelated concurrent tasks are restricted too."
    A long-lived restricted scope throttles the whole process for as long as it is open. This
    is deliberate — the alternative direction (letting the widest request win) oversubscribes,
    which is the problem this package exists to solve — but it is worth knowing before opening
    a budget scope around a long-running operation.

## All-or-nothing libraries

Polyester and NFFT are switched **off whenever the applied budget is below
[`NestedThreading.capacity`](@ref)**, not only when it is 1: a budget of 4 inside an
8-thread outer loop still means the machine is fully occupied, and a nested Polyester loop
spawning its own workers on top of that is exactly the oversubscription being avoided. NFFT
has no partial control at all; Polyester does, and it is deliberately not used — see the
measurements below.

The one exception is a loop that *is* the Polyester consumer. [`@budgeted_batch`](@ref) passes
`exclude = (:polyester,)` so that the `@batch` loop being budgeted is not disabled by its own
restriction.

!!! warning "Measure in the regime you deploy in"
    Whether budgeting helps depends almost entirely on how `Threads.nthreads()` compares to
    the machine's core count, and the difference is not a small factor. Both benchmarks from
    `test/runtests.jl :benchmark`, run on the same 48-core machine at two thread counts:

    | benchmark | `nthreads()` | unbudgeted | budgeted | |
    |---|---|---|---|---|
    | `nt×GEMM(384)` | 8 | 0.0280 s | 0.0190 s | 1.5x |
    | `nt×GEMM(384)` | 48 | **85.998 s** | **0.102 s** | **846x** |
    | `nt×(@batch over 16384)` | 8 | 0.00256 s | 0.00405 s | 0.63x — *loses* |
    | `nt×(@batch over 16384)` | 48 | **1.103 s** | **0.0135 s** | **82x** |

    At `nthreads()` well below the core count there is spare hardware, nesting is free, and
    restricting only wastes threads — the Polyester row actually comes out *negative* there.
    At `nthreads()` near the core count the unbudgeted versions collapse: 86 seconds for
    work that takes a tenth of a second. Benchmarks run in the first regime will tell you
    this package is useless, and that conclusion does not transfer.

    `BenchHelpers.regime_warning` in the test suite flags the mismatch when it sees one.

!!! note "Why guarded pools are all-or-nothing"
    PolyesterWeave can reserve *some* worker slots, so an obvious refinement is to reserve
    only the `capacity() ÷ budget` the outer loop is using and leave the rest for nested
    loops. That was implemented and measured: 2.7x better at `nthreads() = 8` with 2 outer
    iterations, and **320x worse** at `nthreads() = 48` with 2 outer iterations. At high
    thread counts a nested parallel loop is catastrophic however few workers it gets, so
    leaving any free reopens exactly the pathology being closed. All-or-nothing costs a
    bounded factor in the under-subscribed regime and avoids an unbounded one in the
    saturated regime, so that is what the Polyester guard does.

!!! note "Budget scopes themselves are free"
    The benchmark suite asserts this rather than claiming it. Against a plain
    `Threads.@threads` loop, three controls all come out at 0.99-1.00x: a do-nothing closure
    wrapper, a budget scope with the `:polyester` pool excluded, and the Polyester guard
    wrapped around a loop with no `@batch` in it at all. Whatever budgeting costs or saves,
    none of it is overhead from this package.

    Numbers from `julia --project=test test/runtests.jl :benchmark`; rerun them on your own
    machine and thread count. The benchmarks use BenchmarkTools with *paired* measurements —
    baseline and variant back-to-back within each round, median of the per-round ratios —
    because on a dual-socket machine the same loop is bimodal depending on how threads land
    across sockets, by far more than the effects being measured. Paired, the control numbers
    reproduce to within 1% run to run.

## Worked example

With `Threads.nthreads() == 8`:

| loop | trip count | budget | BLAS/FFTW inside | Polyester/NFFT inside |
| --- | --- | --- | --- | --- |
| `@budgeted_threads for i in 1:32` | 32 | `max(1, 8 ÷ 32)` = 1 | 1 | off |
| `@budgeted_threads for i in 1:2` | 2 | `8 ÷ 2` = 4 | 4 | off |
| `@budgeted_threads for i in 1:1` | 1 | `8 ÷ 1` = 8 | 8 | on |
| `@budgeted_threads threads = false for i in 1:2` | — | 1 | 1 | off |
