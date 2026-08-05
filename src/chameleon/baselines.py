"""Fixed-frame baselines (checkerboard and local-rule masks)."""
import sys
import numpy as np
from ._root import ROOT

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import chameleon.vendor.multicode_deform as MC


def masks_bin(C, rX0, rZ0):
    """The four fixed binary frames as int vectors: CSS / XZZX / ZXXZ / Tiurev."""
    n = C["n"]
    return {"CSS": np.zeros(n, int),
            "XZZX": np.array(MC.m_xzzx(C, rX0, rZ0), int),
            "ZXXZ": np.array(MC.m_zxxz(C, rX0, rZ0), int),
            "Tiurev": np.array(MC.m_tiurev(C, rX0, rZ0), int)}
