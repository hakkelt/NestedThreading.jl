# API reference

```@docs
NestedThreading
```

## Loop macros

```@docs
@budgeted
@budgeted_threads
@budgeted_batch
```

## Scoped budgets

```@docs
with_restricted_threads
with_full_threads
enable_full_threading
NestedThreading.with_thread_budget
NestedThreading.with_thread_default
NestedThreading.with_thread_grant
```

## Registry

```@docs
NestedThreading.register_counted_pool!
NestedThreading.register_guarded_pool!
NestedThreading.register_quiesce_pool!
NestedThreading.register_park_hook!
NestedThreading.CountedPool
NestedThreading.GuardedPool
NestedThreading.QuiescePool
NestedThreading.ParkHook
NestedThreading.park_openblas
NestedThreading.capacity
NestedThreading.budget_for
```

## Releasing Julia's threads

```@docs
quiesce_foreign_pools
```
