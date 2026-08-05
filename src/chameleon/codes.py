"""Code constructors (verbatim from final_protocol.py build_color/get_code, delegating to
the vendored modules for the surface and BB constructions)."""
import sys
import numpy as np
from ._root import ROOT

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import chameleon.vendor.multicode_deform as MC   # surface codes + fixed-frame masks
import chameleon.vendor.qldpc_deform as Q        # BB codes

from ._deps import required
with required("panqec", purpose="the Color666 planar-code construction"):
    from panqec.codes import Color666PlanarCode

COLR = {"red": 0, "green": 1, "blue": 2}


def build_color(size):
    """Build the distance-``size`` 6.6.6 planar color code.

    Returns the same (checks, X-logicals, Z-logicals) shape as the other code
    builders, plus the per-stabilizer colors Chromobius needs to decode it.
    """
    code = Color666PlanarCode(size); n = code.n
    qc = list(code.qubit_coordinates); qidx = {c: i for i, c in enumerate(qc)}
    rowsX, rowsZ = [], []
    for sc in code.stabilizer_coordinates:
        _, col, pauli = code.stabilizer_type(sc).split("-")
        sup = [qidx[q] for q in code.get_stabilizer(sc)]
        (rowsX if pauli == "x" else rowsZ).append((sup, COLR[col], sc))
    def to_H(rows):
        H = np.zeros((len(rows), n), np.uint8)
        for i, (sup, _, _) in enumerate(rows):
            H[i, sup] = 1
        return H
    C = dict(n=n, coords=qc, Hx=to_H(rowsX), Hz=to_H(rowsZ),
             LxX=np.zeros((1, n), np.uint8), LzZ=np.zeros((1, n), np.uint8))
    for q in code.get_logicals_x()[0]:
        C["LxX"][0, qidx[q]] = 1
    for q in code.get_logicals_z()[0]:
        C["LzZ"][0, qidx[q]] = 1
    return C, rowsX, rowsZ


def get_code(spec):
    """Build a code from its spec string.

    spec is "surf2d:<odd d>", "color2d:<odd d>" or "BB<n>" (e.g. "surf2d:5",
    "color2d:7", "BB72"). Returns (C, rowsX, rowsZ); the row lists are None
    except for color codes, where the decoder needs the plaquette colouring.

    Raises ValueError naming the valid options rather than letting a typo
    surface as a KeyError from a lookup table three modules down.
    """
    if not isinstance(spec, str) or not spec:
        raise ValueError("code spec must be a non-empty string, got %r" % (spec,))

    if spec.startswith(("surf2d:", "color2d:")):
        fam, _, dtxt = spec.partition(":")
        try:
            d = int(dtxt)
        except ValueError:
            raise ValueError("bad distance in code spec %r: %r is not an integer"
                             % (spec, dtxt)) from None
        if d < 3 or d % 2 == 0:
            raise ValueError("bad distance in code spec %r: d must be an odd "
                             "integer >= 3, got %d" % (spec, d))
        if fam == "color2d":
            return build_color((d - 1) // 2)
        return MC.get_code(fam, d), None, None

    if spec.startswith("BB"):
        try:
            return Q.get_bb(spec), None, None
        except KeyError:
            raise ValueError("unknown bivariate-bicycle code %r; available: %s"
                             % (spec, ", ".join(sorted(Q.SPECS)))) from None

    raise ValueError('unknown code spec %r; expected "surf2d:<odd d>", '
                     '"color2d:<odd d>" or one of %s'
                     % (spec, ", ".join(sorted(Q.SPECS))))
