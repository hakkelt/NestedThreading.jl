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

Polyester and NFFT have no partial thread count. They are switched **off whenever the applied
budget is below [`NestedThreading.capacity`](@ref)**, not only when it is 1: a budget of 4
inside an 8-thread outer loop still means the machine is fully occupied, and a nested
Polyester loop spawning its own workers on top of that is exactly the oversubscription being
avoided.

The one exception is a loop that *is* the Polyester consumer. [`@budgeted_batch`](@ref) passes
`exclude = (:polyester,)` so that the `@batch` loop being budgeted is not disabled by its own
restriction.

## Worked example

With `Threads.nthreads() == 8`:

| loop | trip count | budget | BLAS/FFTW inside | Polyester/NFFT inside |
| --- | --- | --- | --- | --- |
| `@budgeted_threads for i in 1:32` | 32 | `max(1, 8 ÷ 32)` = 1 | 1 | off |
| `@budgeted_threads for i in 1:2` | 2 | `8 ÷ 2` = 4 | 4 | off |
| `@budgeted_threads for i in 1:1` | 1 | `8 ÷ 1` = 8 | 8 | on |
| `@budgeted_threads threads = false for i in 1:2` | — | 1 | 1 | off |
