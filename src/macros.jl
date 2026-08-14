# Loop macros.
#
# Design note: every macro here wraps *the loop*, not the loop body. One budget scope per
# loop, entered before the parallel region opens — which is both the cheap thing (a lock
# acquire and a `set_num_threads` per `mul!`, not per iteration) and the correct thing
# (entering a scope concurrently from every worker of the loop it is supposed to be
# budgeting is exactly the interleaving the refcounted registry exists to avoid).

const _MACRO_ERROR = "expected a `for` loop, optionally wrapped in macros"

# Split a macro chain (`@inbounds @simd for ...`) into the chain and the innermost `for`.
function _peel_macros(ex)
    chain = Any[]
    while ex isa Expr && ex.head === :macrocall
        idx = findlast(a -> a isa Expr && (a.head === :for || a.head === :macrocall), ex.args)
        idx === nothing && error("NestedThreading: $_MACRO_ERROR, got `$ex`")
        push!(chain, (ex, idx))
        ex = ex.args[idx]
    end
    (ex isa Expr && ex.head === :for) || error("NestedThreading: $_MACRO_ERROR, got `$ex`")
    return chain, ex
end

# Rebuild a chain returned by `_peel_macros` around `inner`.
function _rewrap(chain, inner)
    for (call, idx) in Iterators.reverse(chain)
        new = copy(call)
        new.args[idx] = inner
        inner = new
    end
    return inner
end

# Destructure `for var in range` (rejecting multi-spec loops, whose trip count is ambiguous).
function _split_for(loop::Expr)
    spec = loop.args[1]
    if !(spec isa Expr && spec.head === :(=))
        error(
            "NestedThreading: only a single `for x in range` iteration spec is supported, " *
                "got `$spec`. Split nested iterations into an inner loop inside the body."
        )
    end
    return spec.args[1], spec.args[2], loop.args[2]
end

_macro_name(ex) = ex isa Expr && ex.head === :macrocall ? string(ex.args[1]) : ""

# Polyester's `@batch` must not be disabled by the very scope that budgets it.
function _auto_exclude(chain)
    for (call, _) in chain
        occursin("@batch", _macro_name(call)) && return (:polyester,)
    end
    return ()
end

function _budgeted_expr(loop::Expr, chain, exclude::Tuple)
    var, range, body = _split_for(loop)
    rangevar, budgetvar = gensym("range"), gensym("budget")
    parallel = _rewrap(chain, Expr(:for, Expr(:(=), var, rangevar), body))
    return quote
        let $rangevar = $range,
                $budgetvar = $(GlobalRef(NestedThreading, :budget_for))($rangevar)
            $(GlobalRef(NestedThreading, :with_thread_budget))(
                $budgetvar; exclude = $exclude
            ) do
                $parallel
            end
        end
    end
end

function _switched_expr(cond, loop::Expr, chain, mkparallel, exclude::Tuple)
    var, range, body = _split_for(loop)
    rangevar, budgetvar = gensym("range"), gensym("budget")
    sequential = _rewrap(chain, Expr(:for, Expr(:(=), var, rangevar), body))
    parallel = _rewrap(chain, mkparallel(Expr(:for, Expr(:(=), var, rangevar), body)))
    return quote
        let $rangevar = $range
            if $cond
                let $budgetvar = $(GlobalRef(NestedThreading, :budget_for))($rangevar)
                    $(GlobalRef(NestedThreading, :with_thread_budget))(
                        $budgetvar; exclude = $exclude
                    ) do
                        $parallel
                    end
                end
            else
                $(GlobalRef(NestedThreading, :with_restricted_threads))() do
                    $sequential
                end
            end
        end
    end
end

# `@budgeted_threads [threads = expr] for ... end`
function _parse_switched(args, macroname)
    cond = true
    loop = nothing
    for arg in args
        if arg isa Expr && arg.head === :(=) && arg.args[1] === :threads
            cond = arg.args[2]
        elseif arg isa Expr && (arg.head === :for || arg.head === :macrocall)
            loop = arg
        elseif !(arg isa LineNumberNode)
            error("NestedThreading: `$macroname` got unexpected argument `$arg`")
        end
    end
    loop === nothing && error("NestedThreading: `$macroname` $_MACRO_ERROR")
    return cond, loop
end

"""
    @budgeted <parallel-loop>

Wrap an already-chosen parallel loop so that its body runs with inner threaded libraries
limited to `budget_for(range)` threads. The loop construct is emitted unchanged — this
macro never decides *whether* to parallelize, only what budget the workers get.

```julia
@budgeted Threads.@threads for j in 1:num_threads
    mul!(view(out, :, j), op, view(inp, :, j))
end
```

Accepts any chain of macros ending in a single `for x in range` loop, so
`Threads.@threads :static for`, `@inbounds Polyester.@batch for`, and
`@batch minbatch=64 for` all work. The range expression is hoisted into a temporary and
evaluated exactly once. A chain containing `@batch` automatically excludes the `:polyester`
pool, since there the loop being budgeted is itself the Polyester consumer.

Short alias: `@nt`. See also [`@budgeted_threads`](@ref), [`@budgeted_batch`](@ref).
"""
macro budgeted(ex)
    chain, loop = _peel_macros(ex)
    isempty(chain) && error(
        "NestedThreading: `@budgeted` expects a parallel loop construct " *
            "(e.g. `@budgeted Threads.@threads for ...`); use `@budgeted_threads` or " *
            "`@budgeted_batch` for a bare `for` loop."
    )
    return esc(_budgeted_expr(loop, chain, _auto_exclude(chain)))
end

"""
    @budgeted_threads [threads = <bool>] for x in range ... end

`Threads.@threads` over `range` with an automatic inner thread budget, plus a runtime
switch for whether to parallelize at all.

* `threads = true` (the default when omitted) runs `Threads.@threads` with the body's
  inner libraries limited to `budget_for(range)`.
* `threads = false` runs a plain sequential loop with inner libraries limited to a single
  thread, on the assumption that a caller who switched threading off did so because
  concurrency is happening somewhere else.

Short alias: `@ntt`.
"""
macro budgeted_threads(args...)
    cond, ex = _parse_switched(args, "@budgeted_threads")
    chain, loop = _peel_macros(ex)
    mk = body -> Expr(
        :macrocall,
        Expr(:., Expr(:., :Base, QuoteNode(:Threads)), QuoteNode(Symbol("@threads"))),
        __source__,
        body,
    )
    return esc(_switched_expr(cond, loop, chain, mk, ()))
end

"""
    @budgeted_batch [threads = <bool>] for x in range ... end

Polyester `@batch` over `range` with an automatic inner thread budget and the same
`threads =` switch as [`@budgeted_threads`](@ref).

The calling module must have `Polyester` available (`import Polyester`), since the
expansion emits `Polyester.@batch`. The `:polyester` pool is excluded from the budget scope
so that the generated `@batch` loop is not disabled by its own restriction.

Short alias: `@ntb`.
"""
macro budgeted_batch(args...)
    cond, ex = _parse_switched(args, "@budgeted_batch")
    chain, loop = _peel_macros(ex)
    mk = body -> Expr(
        :macrocall,
        Expr(:., :Polyester, QuoteNode(Symbol("@batch"))),
        __source__,
        body,
    )
    return esc(_switched_expr(cond, loop, chain, mk, (:polyester,)))
end

"""
    @nt

Short alias for [`@budgeted`](@ref).
"""
const var"@nt" = var"@budgeted"

"""
    @ntt

Short alias for [`@budgeted_threads`](@ref).
"""
const var"@ntt" = var"@budgeted_threads"

"""
    @ntb

Short alias for [`@budgeted_batch`](@ref).
"""
const var"@ntb" = var"@budgeted_batch"
