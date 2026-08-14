using TestItemRunner

# Usage:
#   julia --project=test test/runtests.jl              # everything except :benchmark
#   julia --project=test test/runtests.jl :macros      # only items tagged :macros
#   julia --project=test test/runtests.jl :benchmark   # the local-only benchmarks
#   julia --project=test test/runtests.jl "@budgeted_batch does not disable its own loop"
#
# Tags: :registry, :macros, :extensions, :jet, :benchmark.
#
# The benchmark items are excluded by default: they compare wall-clock times and so are
# meaningful only on an otherwise idle machine, not in CI.
#
# Test items are run sequentially on purpose — several of them observe process-global
# thread-count state through a probe pool, which parallel execution would interleave.

const FILTER_PARTS = if length(ARGS) > 0
    @assert length(ARGS) == 1
    split(ARGS[1], ",")
else
    String[]
end
const FILTER_TAGS = map(p -> Symbol(p[2:end]), filter(x -> startswith(x, ":"), FILTER_PARTS))
const FILTER_NAMES = filter(x -> !startswith(x, ":"), FILTER_PARTS)

const VERB = get(ENV, "NESTEDTHREADING_TEST_VERBOSE", "false") == "true"

const FILTER = if length(FILTER_PARTS) > 0
    ti -> begin
        run_item = any(t -> t in ti.tags, FILTER_TAGS) || any(n -> n == ti.name, FILTER_NAMES)
        VERB && run_item && println("Running @testitem: ", ti.name)
        run_item
    end
else
    ti -> begin
        run_item = !(:benchmark in ti.tags)
        VERB && run_item && println("Running @testitem: ", ti.name)
        run_item
    end
end

@run_package_tests filter = FILTER
