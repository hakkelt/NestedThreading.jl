module NestedThreadingMKLExt

import LinearAlgebra
import MKL
import NestedThreading

# `LinearAlgebra.BLAS.set_num_threads` goes through libblastrampoline, which forwards to
# MKL's native `mkl_set_num_threads` — except the two do not reliably stay in sync: setting
# one does not always update what the other reports back, see
# https://github.com/JuliaLinearAlgebra/MKL.jl/issues/174. So MKL gets its own pool, driven
# directly through `MKL.get_num_threads`/`MKL.set_num_threads` (MKL's native
# `mkl_get_max_threads`/`mkl_set_num_threads`), registered alongside the always-on `:blas`
# pool rather than instead of it, so both knobs are always set together.
_get() = Int(MKL.get_num_threads())
_set(n::Int) = MKL.set_num_threads(LinearAlgebra.BlasInt(n))

function __init__()
    return NestedThreading.register_counted_pool!(_get, _set; name = :mkl)
end

end # module NestedThreadingMKLExt
