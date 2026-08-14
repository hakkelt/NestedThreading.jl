# API reference

## Loop macros

```@docs
@budgeted
@budgeted_threads
@budgeted_batch
@nt
@ntt
@ntb
```

## Scoped budgets

```@docs
with_restricted_threads
with_full_threads
enable_full_threading
NestedThreading.with_thread_budget
```

## Registry

```@docs
NestedThreading.register_counted_pool!
NestedThreading.register_guarded_pool!
NestedThreading.CountedPool
NestedThreading.GuardedPool
NestedThreading.capacity
NestedThreading.budget_for
```
