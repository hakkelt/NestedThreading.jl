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
```

## Registry

```@docs
NestedThreading.register_counted_pool!
NestedThreading.register_guarded_pool!
NestedThreading.register_quiesce_pool!
NestedThreading.CountedPool
NestedThreading.GuardedPool
NestedThreading.QuiescePool
NestedThreading.capacity
NestedThreading.budget_for
```

## Releasing Julia's threads

```@docs
quiesce_foreign_pools
```
