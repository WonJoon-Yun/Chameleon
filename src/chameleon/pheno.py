"""Phenomenological circuits and DEM utilities (verbatim from pheno_full_suite.build_pheno /
dem_matrices and pheno_banc_suite.build_banc / add_colors)."""
import numpy as np
from ._deps import required
with required("stim", purpose="building the phenomenological circuits"):
    import stim
from .codes import get_code
from .fields import PERMS

DEE = {"surf2d:3": 3, "surf2d:5": 5, "surf2d:7": 7,
       "color2d:3": 3, "color2d:5": 5, "color2d:7": 7,
       "BB18": 4, "BB30": 6, "BB36": 6, "BB72": 6, "BB108": 10, "BB144": 12}


def conj1(v, f):
    """Conjugate one qubit's (pX, pY, pZ) triple by S3 frame ``f``.

    A Clifford frame permutes which physical Pauli presents as which error to the
    code; the rates are unchanged, only relabelled. This is the whole mechanism
    the protocol optimizes over.
    """
    return [v[PERMS[f][0]], v[PERMS[f][1]], v[PERMS[f][2]]]


def build_pheno(spec, pX, pY, pZ, q, F, mem, T=None, g=0.0):
    """T=d rounds (or T rounds if given), per-round data Pauli channel (frame-conjugated),
    uniform MPP flip q. T=1, q=0 gives the code-capacity model (one data-error round,
    perfect syndrome) used by the surrogate-vs-LER diagnostic.
    g: per-round symmetric DEPOLARIZE1 'gate-noise' on the data qubits (default 0, no-op). It is
    frame-INVARIANT (depolarizing commutes with every S_3 axis relabeling), so the frame cannot re-price
    it -- a sensitivity knob for how much residual gate/circuit noise the frame-selection gain survives."""
    C, _, _ = get_code(spec); n = C["n"]; T = DEE[spec] if T is None else T
    HX, HZ = C["Hx"], C["Hz"]; nx = HX.shape[0]
    P3 = np.stack([pX, pY, pZ], 1)
    c = stim.Circuit(); dq = list(range(n))
    if mem == "x":
        c.append("H", dq)
    checks = []
    for i in range(nx):
        checks.append(("X", i, np.nonzero(HX[i])[0]))
    for i in range(HZ.shape[0]):
        checks.append(("Z", i, np.nonzero(HZ[i])[0]))
    mrec = []; nmeas = 0
    for t in range(T):
        for qb in dq:
            v = conj1(list(P3[qb]), int(F[qb]))
            v = [max(min(x, 0.32), 0.0) for x in v]
            if sum(v) > 0:
                if min(v) == 0.0:
                    # a zero component makes the channel non-decomposable into independent
                    # flips; realize it as independent per-axis errors (exact for the nonzero
                    # axes, differs only by the O(p^2) cross term). Zero components are floored
                    # at 1e-9 so every mechanism type exists in the decoder's error model:
                    # Chromobius cannot explain detection patterns whose edges are absent.
                    for gate, pv in (("X_ERROR", v[0]), ("Y_ERROR", v[1]), ("Z_ERROR", v[2])):
                        c.append(gate, [qb], max(pv, 1e-9))
                else:
                    c.append("PAULI_CHANNEL_1", [qb], v)
        if g > 0:
            c.append("DEPOLARIZE1", dq, g)   # frame-invariant gate-noise proxy (symmetric; not re-priceable)
        for kind, i, sup in checks:
            tg = []
            for j, qi in enumerate(sup):
                if j:
                    tg.append(stim.target_combiner())
                tg.append(stim.target_x(int(qi)) if kind == "X" else stim.target_z(int(qi)))
            c.append("MPP", tg, q)
            mrec.append((kind, i, t)); nmeas += 1
    idx = {}
    for m, (kind, i, t) in enumerate(mrec):
        idx[(kind, i, t)] = m - nmeas
    det_kind = "Z" if mem == "z" else "X"
    for t in range(T):
        for kind, i, _ in checks:
            if t == 0:
                if kind != det_kind:
                    continue
                c.append("DETECTOR", [stim.target_rec(idx[(kind, i, 0)])])
            else:
                c.append("DETECTOR", [stim.target_rec(idx[(kind, i, t)]),
                                      stim.target_rec(idx[(kind, i, t - 1)])])
    if mem == "x":
        c.append("H", dq)
    c.append("M", dq)
    dpos = {qb: -(n - j) for j, qb in enumerate(dq)}
    Hm = HZ if mem == "z" else HX
    Lg = C["LzZ"] if mem == "z" else C["LxX"]
    out = stim.Circuit(str(c))
    for i in range(Hm.shape[0]):
        tg = [stim.target_rec(dpos[qb]) for qb in np.nonzero(Hm[i])[0]]
        tg.append(stim.target_rec(idx[(det_kind, i, T - 1)] - n))
        out.append("DETECTOR", tg)
    for li in range(Lg.shape[0]):
        tg = [stim.target_rec(dpos[qb]) for qb in np.nonzero(Lg[li])[0]]
        out.append("OBSERVABLE_INCLUDE", tg, li)
    return out



def add_colors(spec, circ, mem, T=None):
    """Rewrite DETECTORs with (x, y, t, color) coordinates for Chromobius.
    T must match the round count of `circ` (T=1 for the code-capacity build); the
    default DEE[spec] is the phenomenological T=d."""
    C, rowsX, rowsZ = get_code(spec)
    nx = C["Hx"].shape[0]; T = DEE[spec] if T is None else T
    det_kind = "Z" if mem == "z" else "X"
    checks = [("X", i) for i in range(nx)] + [("Z", i) for i in range(C["Hz"].shape[0])]
    coords = []
    for t in range(T):
        for kind, i in checks:
            if t == 0 and kind != det_kind:
                continue
            sup, col, sc = (rowsX[i] if kind == "X" else rowsZ[i])
            coords.append((sc[0], sc[1], t, col + (0 if kind == "X" else 3)))
    rowsM = rowsZ if mem == "z" else rowsX
    for i in range(len(rowsM)):
        sup, col, sc = rowsM[i]
        coords.append((sc[0], sc[1], T, col + (3 if mem == "z" else 0)))
    out = stim.Circuit(); ci = 0
    for inst in circ.flattened():
        if inst.name == "DETECTOR":
            out.append("DETECTOR", inst.targets_copy(), list(map(float, coords[ci]))); ci += 1
        else:
            out.append(inst)
    if ci != len(coords):   # raise (not assert): must survive python -O
        raise RuntimeError("pheno: detector-coordinate misalignment (%d != %d)" % (ci, len(coords)))
    return out


def dem_matrices(circ):
    """Flatten a circuit's detector error model into (H, O, priors) for BP+OSD.

    ``H`` maps each independent error mechanism to the detectors it fires, ``O``
    to the observables it flips, and ``priors`` carries their probabilities. Built
    without error decomposition, since the qLDPC decoding graph is not graphlike
    and BP+OSD -- unlike matching -- does not need it to be.
    """
    dem = circ.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    nd = dem.num_detectors; no = dem.num_observables
    rows = []; obs = []; pr = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]; ds = []; os_ = []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                ds.append(t.val)
            elif t.is_logical_observable_id():
                os_.append(t.val)
        rows.append(ds); obs.append(os_); pr.append(p)
    ne = len(rows)
    H = np.zeros((nd, ne), dtype=np.uint8); O = np.zeros((no, ne), dtype=np.uint8)
    for j, (ds, os_) in enumerate(zip(rows, obs)):
        for d in ds:
            H[d, j] ^= 1
        for o in os_:
            O[o, j] ^= 1
    return H, O, np.array(pr)
