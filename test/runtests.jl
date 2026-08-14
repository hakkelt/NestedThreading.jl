using Test
using NestedThreading
using LinearAlgebra: BLAS

const NT = NestedThreading

# The extension-dependent parts run only when the weakdeps are actually installed, so the
# file stays usable in a bare (LinearAlgebra-only) environment.
const HAS_FFTW = Base.identify_package("FFTW") !== nothing
const HAS_POLYESTER = Base.identify_package("Polyester") !== nothing
const HAS_NFFT = Base.identify_package("NFFT") !== nothing

HAS_FFTW && @eval using FFTW
HAS_POLYESTER && @eval import Polyester
HAS_NFFT && @eval import NFFT

include("registry_tests.jl")
include("macro_tests.jl")
include("extension_tests.jl")
include("jet_tests.jl")
