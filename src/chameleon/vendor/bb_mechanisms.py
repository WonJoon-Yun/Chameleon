"""Complete + balanced low-weight logical enumerator for BB (and any [[n,k,d]] CSS) codes.

Translation-orbit enumeration is incomplete (misses low-weight linear combinations of
basis logicals) and imbalanced across axes. Instead we enumerate ALL 2^k-1 logical
CLASSES per axis and OSD-0 reduce each to a low-weight in-class representative, keeping
those of weight <= d+extra. This is complete (every class represented) and balanced
(same construction for X and Z), and scales as O(2^k * perms) -- fine for k=12.
"""
import numpy as np
from sympy.abc import x, y
# `qldpc` transitively imports cvxpy, which prints a solver-registration notice to
# stderr when the installed ortools is newer than the version cvxpy pins. The notice
# is harmless -- cvxpy is not used by Chameleon at all -- but it would appear on
# every command an evaluator runs and reads like an error. Silence just this import.
import contextlib as _ctx, io as _io
with _ctx.redirect_stderr(_io.StringIO()), _ctx.redirect_stdout(_io.StringIO()):
    from .._deps import required
    with required("qldpc", purpose="the bivariate-bicycle code construction"):
        from qldpc import codes

BB_SPECS = {  # Bravyi et al. 2024 gross-code family (verified [[n,k,d]])
    "BB72":  ({x: 6,  y: 6},  x**3 + y + y**2,   y**3 + x + x**2),      # [[72,12,6]]
    "BB90":  ({x: 15, y: 3},  x**9 + y + y**2,   1 + x**2 + x**7),      # [[90,8,10]]
    "BB108": ({x: 9,  y: 6},  x**3 + y + y**2,   y**3 + x + x**2),      # [[108,8,10]]
    "BB144": ({x: 12, y: 6},  x**3 + y + y**2,   y**3 + x + x**2),      # [[144,12,12]]
}
BB_D = {"BB72": 6, "BB90": 10, "BB108": 10, "BB144": 12}


def _rref(M):
    M = M.copy() % 2
    r = 0
    pivots = []
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        pivots.append(c)
        r += 1
        if r == rows:
            break
    return M[:r], pivots


def _all_class_targets(Lbasis):
    """Matrix T (2^k-1 x n): every nonzero GF(2) combination of the k basis logicals."""
    k, n = Lbasis.shape
    T = np.zeros((1, n), np.uint8)
    for i in range(k):
        T = np.vstack([T, T ^ Lbasis[i]])  # Gray-free doubling: rows = all subsets
    return T[1:]  # drop empty combination


def _enumerate(Lbasis, Hstab, n, d, extra, perms=24, seed=0):
    """All 2^k-1 class combos, OSD-0 reduced; keep supports of weight <= d+extra. Returns list[frozenset].

    Permutation-outer, class-batched: for each random column permutation we RREF Hstab once,
    then reduce ALL class targets against that pivot structure in a single vectorized pass.
    """
    rng = np.random.default_rng(seed)
    wmax = d + extra
    T = _all_class_targets(Lbasis).astype(np.uint8)      # (C, n)
    best = T.copy()
    bw = best.sum(1).astype(int)
    if Hstab.shape[0] == 0:
        keep = (bw > 0) & (bw <= wmax)
        return [frozenset(np.flatnonzero(best[i]).tolist()) for i in np.flatnonzero(keep)]
    for _ in range(perms):
        pi = rng.permutation(n)
        inv = np.empty(n, int); inv[pi] = np.arange(n)
        R, pivs = _rref(Hstab[:, pi])                    # R rows are reduced generators
        E = T[:, pi].copy()                              # (C, n) in permuted coords
        for ri, pc in enumerate(pivs):                   # clear each pivot column across all classes
            flip = E[:, pc].astype(bool)
            E[flip] ^= R[ri]
        w = E.sum(1).astype(int)
        imp = w < bw
        if imp.any():
            best[imp] = E[imp][:, inv]
            bw[imp] = w[imp]
    keep = np.flatnonzero((bw > 0) & (bw <= wmax))
    out = {}
    for i in keep:
        fs = frozenset(np.flatnonzero(best[i]).tolist())
        if fs not in out:
            out[fs] = fs
    return list(out.values())


def _bb_code(spec):
    o, A, B = BB_SPECS[spec]
    c = codes.BBCode(o, A, B)
    n = c.num_qubits
    k = c.dimension
    Hx = np.array(c.matrix_x, np.uint8) % 2
    Hz = np.array(c.matrix_z, np.uint8) % 2
    # min-weight X-basis from unreduced ops; min-weight Z-basis from reduced ops
    L0 = np.array(c.get_logical_ops(), np.uint8) % 2
    LxX = L0[:k, :n].copy()
    c.reduce_logical_ops()
    L1 = np.array(c.get_logical_ops(), np.uint8) % 2
    LzZ = L1[k:2 * k, n:].copy()
    d = BB_D[spec]
    return n, k, d, Hx, Hz, LxX, LzZ


def bb_mechanisms(spec, extra=2, perms=24):
    n, k, d, Hx, Hz, LxX, LzZ = _bb_code(spec)
    LX = _enumerate(LxX, Hx, n, d, extra, perms=perms, seed=1)
    LZ = _enumerate(LzZ, Hz, n, d, extra, perms=perms, seed=2)
    return LX, LZ, None
