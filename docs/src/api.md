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
with_threads_disabled
with_threads_enabled
NestedThreading.reset_thread_counts!
NestedThreading.with_thread_limit
NestedThreading.with_thread_default
NestedThreading.with_thread_grant
NestedThreading.thread_limit
```

## Registry

```@docs
NestedThreading.register_pool!
NestedThreading.capacity
```

## Releasing Julia's threads

```@docs
NestedThreading.quiesce_foreign_pools
```
