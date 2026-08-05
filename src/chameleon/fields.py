"""Calibration pools and per-qubit 3-class noise fields (verbatim from final_protocol.py
pool3/field3 and pheno_full_suite.py field3x / pheno_banc_suite.py fields_ext)."""
import json, math, csv
import numpy as np
from ._root import rpath
from .config import ProtocolConfig
_CFG = ProtocolConfig()

# calibration CSVs (formerly experiments/multicode_deform.CSV)
CSV = {"ibm_berlin": "calibration/ibm_berlin_calibrations_2026-06-17T17_56_32Z.csv",
       "ibm_miami": "calibration/ibm_miami_calibrations_2026-06-16T12_44_22Z.csv"}

PERMS = [(0, 1, 2), (2, 1, 0), (1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1)]

_P3 = {}

# The device pools pool3() knows how to build. Named here so an unknown noise
# field can say what IS available instead of echoing the bad string back.
_DEVICES = ("berlin", "miami", "willow")


def pool3(kind):
    """Per-qubit (pX,pY,pZ) pool for a device: Pauli-twirled T1/T2 idle over the readout
    window + per-CZ twirled marginals split per letter (IBM); T1/T2 twirl only (Willow)."""
    if kind in _P3:
        return _P3[kind]
    if kind == "willow":
        cal = json.load(open(rpath("calibration/willow_calibration.json")))
        arr = []
        for v in cal["per_qubit"].values():
            pxy = (1 - math.exp(-25.0 / v["T1_ns"])) / 4.0
            pz = max((1 - math.exp(-25.0 / v["T2_ns"])) / 2.0 - pxy, 0.0)
            arr.append([pxy, pxy, pz])
        _P3[kind] = np.array(arr)
    else:
        dev = "ibm_" + kind
        T1 = {}; T2 = {}; rd = {}
        for r in csv.DictReader(open(rpath(CSV[dev]))):
            try:
                q = int(r["Qubit"]); T1[q] = float(r["T1 (us)"]); T2[q] = float(r["T2 (us)"])
                rd[q] = float(r["Readout length (ns)"]) / 1000.0
            except ValueError:
                pass      # a blank or malformed cell in the calibration CSV: that
                          # qubit simply contributes no readout time to the pool
        t = float(np.median(list(rd.values())))
        P = {}
        for q in T1:
            pxy = (1 - math.exp(-t / T1[q])) / 4.0
            pz = max((1 - math.exp(-t / T2[q])) / 2.0 - pxy, 0.0)
            P[q] = [pxy, pxy, pz]
        cz = json.load(open(rpath(_CFG.pool_dir, "fakechar_%s.json" % dev)))["couplers"]
        for k, ch in cz.items():
            _, a, b = k.split("_"); a, b = int(a), int(b)
            for pstr, v in ch.items():
                for qi, sym in [(a, pstr[0]), (b, pstr[1])]:
                    if qi in P and sym != "I":
                        P[qi]["XYZ".index(sym)] += v
        _P3[kind] = np.array(list(P.values()))
    return _P3[kind]


def _check_rate(P, noise):
    """A physical error rate must lie in (0, 1).

    Called at both public entry points (field3 and field3x); the synthetic
    families field3x handles itself never reach field3, so one guard is not
    enough.

    Outside that range the surrogate does not fail: gamma clips a negative
    r(1-r) to zero, so an invalid rate reads as a perfect qubit and the score
    comes back finite and optimistic. Reject it here instead.
    """
    if not isinstance(P, (int, float)) or isinstance(P, bool):
        raise ValueError("physical error rate must be a number, got %r" % (P,))
    if not (0.0 < P < 1.0):
        raise ValueError(
            "physical error rate out of range for noise %r: p=%r must satisfy "
            "0 < p < 1 (the protocol grid runs 1e-3 to 1e-2 and never exceeds "
            "0.03)" % (noise, P))


def field3(noise, n, seed, P):
    """n resampled qubit triples, mean total rescaled to P. noise: '<dev>_star' | 'half:<eta>'."""
    _check_rate(P, noise)
    rng = np.random.default_rng(seed)
    if noise.endswith("_staruni"):
        # ablation: device bias DIRECTION kept, but EVERY qubit's total rate set to P (uniform
        # magnitude) instead of mean-rescaled -- isolates magnitude heterogeneity from direction.
        pool = pool3(noise.split("_")[0])
        sel = pool[rng.integers(0, len(pool), n)].astype(float)
        sel = sel * (P / np.maximum(sel.sum(1, keepdims=True), 1e-15))   # PER-QUBIT rescale
        return sel[:, 0], sel[:, 1], sel[:, 2]
    if noise.endswith("_star"):
        pool = pool3(noise.split("_")[0])
        sel = pool[rng.integers(0, len(pool), n)]
        sel = sel * (P / sel.sum(1).mean())
        return sel[:, 0], sel[:, 1], sel[:, 2]
    if noise.startswith("half:"):
        eta = float(noise.split(":")[1])
        hi, lo = P * eta / (eta + 1), P / (eta + 1)
        bx = rng.random(n) < 0.5
        return np.where(bx, hi, lo), np.zeros(n), np.where(bx, lo, hi)
    raise ValueError(
        "unknown noise field %r; expected '<device>_star' or '<device>_staruni' "
        "for a resampled device pool (device in %s), or 'half:<eta>'"
        % (noise, ", ".join(sorted(_DEVICES))))


def field3x(noise, n, seed, P):
    """field3 plus the synthetic three-way-mixed 'xyz:<eta>' class and the
    'drift:<sigma>:<dseed>:<base>' class (base field with every per-qubit axis
    rate multiplied by an independent lognormal factor of width sigma --
    the deployment-time miscalibration model for the drift study)."""
    _check_rate(P, noise)
    if noise.startswith("drift:"):
        _, sig, dseed, base = noise.split(":", 3)
        pX, pY, pZ = field3x(base, n, seed, P)
        rng = np.random.default_rng(int(dseed) * 1000003 + seed + 12345)
        f = np.exp(rng.normal(0.0, float(sig), (3, n)))
        return pX * f[0], pY * f[1], pZ * f[2]
    if noise.startswith("xyz:"):
        eta = float(noise.split(":")[1]); rng = np.random.default_rng(seed)
        hi, lo = P * eta / (eta + 2), P / (eta + 2)
        ax = rng.integers(0, 3, n)
        p = np.full((n, 3), lo); p[np.arange(n), ax] = hi
        return p[:, 0], p[:, 1], p[:, 2]
    if noise.startswith("xyzmix:"):
        # A4 gate-noise-dilution model: fraction f of the total rate p is UNSTRUCTURED
        # uniform depolarizing (un-repriceable "gate noise"), the remaining (1-f) is the
        # biased xyz structure the frame re-prices. f=0 == xyz:eta; f=1 == depolarizing;
        # per-qubit total stays exactly p. Quantifies how fast the gain dilutes as noise
        # the frame cannot re-price grows (adversarial attack A4).
        _, eta, f = noise.split(":"); eta = float(eta); f = float(f)
        rng = np.random.default_rng(seed)
        ps = P * (1.0 - f)                       # structured (re-priceable) budget
        hi, lo = ps * eta / (eta + 2), ps / (eta + 2)
        ax = rng.integers(0, 3, n)
        p = np.full((n, 3), lo); p[np.arange(n), ax] = hi
        p += P * f / 3.0                          # uniform depolarizing floor
        return p[:, 0], p[:, 1], p[:, 2]
    return field3(noise, n, seed, P)


def fields_ext(noise, n, m, fseed, p):
    """n data + m ancilla triples from one construction, jointly mean-rescaled to p."""
    tot = n + m
    pX, pY, pZ = field3x(noise, tot, fseed, p)
    s = (pX + pY + pZ).mean(); sc = p / max(s, 1e-15)
    pX, pY, pZ = pX * sc, pY * sc, pZ * sc
    return (pX[:n], pY[:n], pZ[:n]), (pX[n:], pY[n:], pZ[n:])


def present_s3(P3, F):
    """S3 frame (perm indices per qubit) -> presented (rX, rZ) marginals."""
    F = np.asarray(F, int)
    newp = np.take_along_axis(P3, np.array([list(PERMS[f]) for f in F]), axis=1)
    return newp[:, 0] + newp[:, 1], newp[:, 2] + newp[:, 1]
