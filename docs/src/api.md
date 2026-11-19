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
thread_limit
```

## Registry

```@docs
NestedThreading.register_pool!
NestedThreading.capacity
NestedThreading.budget_for
```

## Releasing Julia's threads

```@docs
quiesce_foreign_pools
```
