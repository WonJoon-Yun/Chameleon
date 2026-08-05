"""Surface and color code constructions, and the fixed-frame baselines.

Two things the rest of the package builds on:

  `get_code`  wraps panqec's rotated-planar and 6.6.6 colour codes into the plain
              dict form (Hx, Hz, logicals, coordinates) everything here expects.
  `m_*`       the fixed baseline frames -- CSS is the identity and needs no
              function, XZZX and ZXXZ are the two phases of a coordinate
              checkerboard, and Tiurev is the local rule that deforms each qubit
              so its dominant error axis lands on a fixed template.

The baselines are FIXED: XZZX and ZXXZ ignore the noise entirely, and Tiurev
reads only which axis dominates per qubit. None of them searches, which is the
point of comparing against them.

This module previously also carried a standalone experiment driver (a candidate
sweep, a BP+OSD code-capacity decoder, a device-noise sampler and a `main`).
None of it was reachable from the artifact, and one part actively misrepresented
the method: its `m_ours` picked the best of a candidate pool that INCLUDED the
four baselines, which is exactly the comparison this work argues against. It
also wrote into `results/` through a relative path. Removed rather than shipped.
"""
import numpy as np

from .._deps import required
with required("panqec", purpose="the surface and color code constructions"):
    from panqec.codes import RotatedPlanar2DCode, Color666PlanarCode, Color3DCode

#: family -> (panqec class, {code distance: the class's own size parameter})
CODES = {"surf2d":  (RotatedPlanar2DCode, {3: 3, 5: 5, 7: 7, 9: 9}),
         "color2d": (Color666PlanarCode,  {3: 1, 5: 2, 7: 3, 9: 4}),
         "color3d": (Color3DCode,         {4: 2})}


def get_code(name, d):
    """Build family ``name`` at distance ``d`` as a plain dict.

    Returns Hx/Hz (dense mod-2 check matrices), the X and Z logical supports
    restricted to their own halves of the symplectic representation, the qubit
    coordinates the checkerboard baselines are defined on, and n.
    """
    cls, dm = CODES[name]
    c = cls(dm[d]); n = c.n
    Hx = np.array(c.Hx.todense(), dtype=np.uint8) % 2
    Hz = np.array(c.Hz.todense(), dtype=np.uint8) % 2
    LxX = np.array(c.logicals_x, dtype=np.uint8)[:, :n] % 2
    LzZ = np.array(c.logicals_z, dtype=np.uint8)[:, n:] % 2
    coords = [tuple(map(int, q)) for q in c.qubit_coordinates]
    return dict(Hx=Hx, Hz=Hz, LxX=LxX, LzZ=LzZ, coords=coords, n=n)


# ---- baseline frames (S = boolean per-qubit Hadamard mask) -------------------

def cb(coords, phase):
    """The coordinate checkerboard at the given ``phase`` (0 or 1)."""
    return np.array([(int(sum(c) // 2) % 2 == phase) for c in coords])


def m_xzzx(C, pX, pZ):
    """XZZX: Hadamard on one checkerboard phase, independent of the noise."""
    return cb(C["coords"], 0)


def m_zxxz(C, pX, pZ):
    """ZXXZ: XZZX's complement -- the other checkerboard phase."""
    return cb(C["coords"], 1)


def m_tiurev(C, pX, pZ):
    """Tiurev's local rule: deform where the dominant axis misses the template.

    Per qubit it reads only which of pX, pZ is larger and flips when that does
    not already match the checkerboard target. Local and greedy -- it never sees
    the code's failure structure, only the per-qubit ordering.
    """
    domX = pX >= pZ
    want = cb(C["coords"], 0)
    return domX != want


#: index i of _TIU_PERMS is the S3 frame sending the original (X,Y,Z) rates to
#: the slots named by the tuple, matching fields.PERMS.
_TIU_PERMS = [(0, 1, 2), (2, 1, 0), (1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1)]
_TIU_PIDX = {p: i for i, p in enumerate(_TIU_PERMS)}


def tiurev_template(coords):
    """The |row - col| mod 4 template the six-frame Tiurev rule targets."""
    return np.array([abs(int(c[0]) - int(c[1])) % 4 == 0 for c in coords])


def m_tiurev6(C, pX, pY, pZ):
    """Tiurev's rule extended from {I,H} to the full six-frame space.

    Per qubit: sort the three rates, put the largest on the X slot where the
    template says so (otherwise on Z), and hide the RAREST in the shared Y slot,
    which is the one both decoding graphs see. Still local -- one qubit's three
    rates decide its frame, with no reference to any other qubit.
    """
    R = np.stack([pX, pY, pZ], 1)
    tmpl = tiurev_template(C["coords"])
    out = np.zeros(len(pX), int)
    for q in range(len(pX)):
        lo, md, hi = (int(a) for a in np.argsort(R[q]))   # rarest, medium, largest axis
        xs, zs = (hi, md) if tmpl[q] else (md, hi)        # largest -> X slot where template True
        out[q] = _TIU_PIDX[(xs, lo, zs)]                  # rarest hidden in the shared Y slot
    return out
