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

## Scopes

```@docs
with_threads_disabled
with_threads_enabled
with_thread_limit
with_thread_default
with_thread_grant
NestedThreading.thread_limit
NestedThreading.reset_thread_counts!
```

## Registry

```@docs
NestedThreading.register_pool!
thread_pool_info
NestedThreading.capacity
```

## Releasing Julia's threads

```@docs
NestedThreading.quiesce_foreign_pools
```
