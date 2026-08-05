"""Process-environment setup shared by the runners.

Each runner fans work out over a multiprocessing Pool, and every worker imports
numpy. Left alone, each worker's BLAS/OpenMP pool sizes itself to the whole
machine, so N workers on a 160-core host ask for N*160 threads and spend their
time contending rather than computing. Capping has to happen BEFORE numpy is
imported -- the thread pools are sized at import -- which is why `cap_threads`
is called from the runner preamble rather than from library code.

Measured, so it is not mistaken for a correctness fix: the thread count does not
move any result. Selecting frames for surf2d:5, color2d:3 and BB18 at 1 thread
and at 16 gives bit-identical frames and bit-identical surrogate values.
"""
import os

#: Thread-pool variables every BLAS/OpenMP build we ship against honours.
_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
         "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

#: Threads per worker. Two rather than one: the decoders get a little parallelism
#: without the workers fighting over cores.
DEFAULT_THREADS = 2


def cap_threads(n=None):
    """Cap each worker's BLAS/OpenMP threads. Call before importing numpy.

    ``n`` defaults to the CHAM_THREADS environment variable, else
    `DEFAULT_THREADS`. Existing settings win, so a caller who has already chosen
    a value on the command line keeps it. Returns the cap in effect.
    """
    if n is None:
        n = os.environ.get("CHAM_THREADS", DEFAULT_THREADS)
    n = max(1, int(n))
    for var in _VARS:
        os.environ.setdefault(var, str(n))
    return n
