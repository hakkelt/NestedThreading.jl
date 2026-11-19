# Changelog

## 0.2.0

This release reshapes the public API around scopes that disable, enable or limit inner
threading, and replaces the four registration functions with one. Old names are removed
without deprecation shims.

### Renamed

| Removed | Replacement |
| --- | --- |
| `with_thread_budget(f, n)` | `with_thread_limit(f, n)` |
| `with_restricted_threads(f)` | `with_threads_disabled(f)` |
| `with_full_threads(f)` | `with_threads_enabled(f)` |
| `enable_full_threading()` | `NestedThreading.reset_thread_counts!()` |
| `register_counted_pool!(get, set; name)` | `NestedThreading.register_pool!(name; get, set)` |
| `register_guarded_pool!(guard; name)` | `NestedThreading.register_pool!(name; guard)` |
| `register_quiesce_pool!(quiesce; name)` | `NestedThreading.register_pool!(name; quiesce)` |
| `register_park_hook!(park; name)` | `NestedThreading.register_pool!(name; get, set, park)` |

A library with several controls passes them all in one `register_pool!` call; a second call
with the same name throws. `park` must come in the same call as `get`/`set`.

### Removed from the public API

`CountedPool`, `GuardedPool`, `QuiescePool`, `ParkHook`, `park_openblas` and `budget_for`
are internal now. Use `thread_pool_info()` instead of reading the registry's internal state.

### Exports

- Newly exported: `with_thread_limit`, `with_thread_default`, `with_thread_grant`,
  `thread_pool_info`.
- No longer exported, still public: `quiesce_foreign_pools`, `thread_limit`.
- Public, not exported: `register_pool!`, `capacity`, `reset_thread_counts!`.

### Changed

- `register_pool!` throws an `ArgumentError` for a name that is already registered, for
  `get` without `set` or the reverse, for `park` without `get`/`set`, for no control at
  all, and when `get()` does not return an `Integer`. Registering a duplicate name used to
  be a silent no-op.
- `thread_limit(name)` returns `nothing` instead of `typemax(Int)` when no enclosing limit
  applies, and always on Julia 1.10.

### Added

- `thread_pool_info()`: every registered library with its controls and its current,
  unrestricted and effective thread counts; `thread_pool_info(name)` gives the entry of one
  library, or `nothing` if none is registered under that name.
- The `exclude` and `only` keywords of every scope function take a single name or any
  collection of names (`exclude = :polyester`, `only = [:blas, :mkl]`), not only a tuple.
  An element that is not a `Symbol` throws an `ArgumentError`.
- `@budgeted_threads` and `@budgeted_batch` pass every argument before the loop other than
  `threads =` to the macro they emit, so a `Threads.@threads` schedule or `@batch` options
  combine with the switch (`@budgeted_batch minbatch = 64 threads = flag for ...`).
- Requests a scope cannot act on (an unknown `exclude`/`only`/`thread_limit` name, a count
  below 1, a held-back `park` hook, and similar) are logged at the `Debug` level; enable
  them with `ENV["JULIA_DEBUG"] = "NestedThreading"`.

### Fixed

- Registering a guard while a scope was open could raise a `BoundsError` when that scope
  closed.
