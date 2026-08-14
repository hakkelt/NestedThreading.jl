module NestedThreadingPolyesterExt

import NestedThreading
import Polyester

# Polyester has no thread-count API: `disable_polyester_threads` reserves every Polyester
# worker for the duration of the call so nested `@batch` loops find none free. That is
# reservation-based rather than save/restore-based, so it nests and composes safely.
_guard(f::F, restricted::Bool) where {F} =
    restricted ? Polyester.disable_polyester_threads(f) : f()

function __init__()
    return NestedThreading.register_guarded_pool!(_guard; name = :polyester)
end

end # module NestedThreadingPolyesterExt
