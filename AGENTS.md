# Notes for coding agents

The README covers what the package does, the API and how to run the tests. This file lists
the rules that are easy to break without noticing.

## Registry concurrency (`src/registry.jl`)

- `POOL_NAMES` and `QUIESCE_POOLS` are read without `REGISTRY_LOCK` (`thread_limit`,
  `_debug_unknown_names`, `quiesce_foreign_pools`). Replace them under the lock, never
  `push!` onto them. All other registry state is touched only with the lock held.
- `SAVED`, `MAXIMA` and `COUNTED_POOLS` have the same length, and index `i` is the same
  pool in all three. Code that adds or removes a pool keeps that true.
- `get`, `set` and `park` hooks run with the lock held, so a `park` hook must not wait on
  a task that opens a scope. `guard` and `quiesce` hooks run without it.
- Nothing logs while the lock is held: hook failures are collected under the lock and
  reported after it is released. Do not add `@debug`/`@warn` inside a locked block.

## Tests

- Test items run sequentially on purpose. They observe process-global state through the
  probe pools in `test/setup.jl`, and registering a name twice throws.
- A test that registers pools checks that the registry is left consistent (names, vector
  lengths, `SAVED`, `MAXIMA`), as `test/test_registry.jl` does.
- Run the suite with `julia -t 1` as well as with several threads: `capacity() == 1` is a
  separate code path.
- Run it on Julia 1.10 too. Without `ScopedValues`, `thread_limit` always returns
  `nothing` and nested hard limits are never skipped.
- The `:benchmark` items compare wall-clock times and mean something only on an idle machine.

## Docs

- Every timing in the docs and docstrings names its CPU, Julia thread count and Julia
  version. When the code it measures changes, measure again and update the label with the
  numbers.
- `docs/make.jl` uses `checkdocs = :exports`: every exported name needs a docstring listed in
  `docs/src/api.md`.

## Library support

Each supported library is an extension in `ext/` that registers itself with one
`NestedThreading.register_pool!` call. A new library follows the same pattern; see
`docs/src/guides/extending.md`.
