module NestedThreadingFFTWExt

import FFTW
import NestedThreading

function __init__()
    return NestedThreading.register_counted_pool!(
        FFTW.get_num_threads, FFTW.set_num_threads; name = :fftw
    )
end

end # module NestedThreadingFFTWExt
