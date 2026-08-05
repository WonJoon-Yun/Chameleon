"""Enriched BB mechanism enumerator: OSD-0 reduction (more perms) + Z_l x Z_m translation-orbit
augmentation. Each low-weight logical's translates are also valid low-weight syndrome-free operators
(BB codes are translation-invariant), which both multiplies the mechanism count and supplies the
degeneracy/multiplicity the union bound needs. Verifies every translate is syndrome-free.
"""
import os as _os
_LR = _os.environ.get("LEVER_ROOT") or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
import sys
sys.path.insert(0, _LR)
import numpy as np
from chameleon.vendor.bb_mechanisms import _bb_code, _all_class_targets, _rref

BB_LM = {"BB72": (6, 6), "BB90": (15, 3), "BB108": (9, 6), "BB144": (12, 6)}  # (l,m); n=2lm


def _reduce_classes(Lbasis, Hstab, n, perms, seed):
    rng = np.random.default_rng(seed)
    T = _all_class_targets(Lbasis).astype(np.uint8)
    best = T.copy(); bw = best.sum(1).astype(int)
    for _ in range(perms):
        pi = rng.permutation(n); inv = np.empty(n, int); inv[pi] = np.arange(n)
        R, pivs = _rref(Hstab[:, pi]); E = T[:, pi].copy()
        for ri, pc in enumerate(pivs):
            flip = E[:, pc].astype(bool); E[flip] ^= R[ri]
        w = E.sum(1).astype(int); imp = w < bw
        if imp.any():
            best[imp] = E[imp][:, inv]; bw[imp] = w[imp]
    return best, bw


def _orbit_vec(v, l, m):
    """All l*m translates of a length-2lm vector, per sector, on the Z_l x Z_m grid."""
    lm = l * m
    out = []
    A = v[:lm].reshape(l, m); B = v[lm:].reshape(l, m)
    for a in range(l):
        for b in range(m):
            RA = np.roll(np.roll(A, a, axis=0), b, axis=1)
            RB = np.roll(np.roll(B, a, axis=0), b, axis=1)
            out.append(np.concatenate([RA.reshape(-1), RB.reshape(-1)]))
    return out


def enrich(spec, extra=2, perms=200, orbit=True, seed0=1):
    n, k, d, Hx, Hz, LxX, LzZ = _bb_code(spec)
    l, m = BB_LM[spec]
    wmax = d + extra
    out = {}
    for axis, (Lb, Hstab, Hchk, sd) in enumerate([(LxX, Hx, Hz, seed0), (LzZ, Hz, Hx, seed0 + 1)]):
        best, bw = _reduce_classes(Lb, Hstab, n, perms, sd)
        reps = [best[i] for i in np.flatnonzero((bw > 0) & (bw <= wmax))]
        acc = {}
        for v in reps:
            cands = _orbit_vec(v, l, m) if orbit else [v]
            for t in cands:
                w = int(t.sum())
                if 0 < w <= wmax and not ((Hchk @ t) % 2).any():   # syndrome-free check
                    fs = frozenset(np.flatnonzero(t).tolist())
                    acc[fs] = fs
        out[axis] = list(acc.values())
    return out[0], out[1]   # LX, LZ


if __name__ == "__main__":
    for spec in ["BB72", "BB144"]:
        LXo, LZo = enrich(spec, extra=2, perms=200, orbit=False)
        LX, LZ = enrich(spec, extra=2, perms=200, orbit=True)
        print("%-6s no-orbit: |LX|=%d |LZ|=%d | +orbit: |LX|=%d |LZ|=%d" % (
            spec, len(LXo), len(LZo), len(LX), len(LZ)), flush=True)
    print("DONE", flush=True)
