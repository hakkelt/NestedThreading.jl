module NestedThreadingFFTWExt

import FFTW
import NestedThreading

function __init__()
    return NestedThreading.register_pool!(
        :fftw; get = FFTW.get_num_threads, set = FFTW.set_num_threads
    )
end

end # module NestedThreadingFFTWExt
