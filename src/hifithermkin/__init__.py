"""hifithermkin: a Gaussian 16 decoder and the RGD1 / stjohn-radicals import."""

import os

# One BLAS thread per process, set before numpy loads. The import runs several
# processes at once, and multithreaded eigensolves of small Hessians were 25x slower
# in the stationary-point validation listener (0.04 s against 0.002 s a point).
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from . import gaussian  # noqa: E402

__version__ = "0.0.0"

__all__ = ["gaussian"]
