"""M1 -- EXACT rare-event (loser) mass vs the Bhattacharyya surrogate U.

Answers the reviewer question the Design section currently asserts without evidence:
  (a) is the EXACT ambiguity objective actually too expensive to sit inside the search?
  (b) how much of that objective does the Bhattacharyya bound preserve?

For a class-c ambiguity operator l with support S(l) and per-qubit presented class-c
marginals x_q, the exact losing mass is
      C_l(F) = sum over the 2^{|l|-1} complementary pattern pairs {e, e^l} of
               min( P[e], P[e^l] )
(off-support qubits factor out of BOTH members of every pair, so the support-restricted
sum IS the exact marginalised quantity -- no approximation in this reduction).
The aggregate objective mirrors U's max-over-classes:
      C(F) = max_c sum_{l in A_c} C_l(F),     U(F) = max_c sum_{l in A_c} prod_q gamma(x_q).

Everything here uses the REPO's own mechanism sets (chameleon.mechs.mechs), the repo's
S3 presentation (fields.present_s3 / PERMS) and the repo's surrogate (surrogate.build_U6).
No re-implementation of the scoring path.

Outputs results/typeA_investigation/m1_exact_vs_bhatt.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

import os, sys, json, time, argparse
import numpy as np

from chameleon._root import rpath
from chameleon.config import ProtocolConfig
_CFG = ProtocolConfig()
from chameleon.codes import get_code
from chameleon.mechs import mechs
from chameleon.fields import field3x, present_s3
from chameleon.surrogate import build_U6, gamma
from chameleon.search import cem_pool, cem6_pool
from chameleon.vendor.multicode_deform import m_tiurev6

OUT = rpath(_CFG.study_dir + "/m1_exact_vs_bhatt.json")

# ---------------------------------------------------------------- exact machinery
_BITS = {}


def half_patterns(w):
    """(2^{w-1}, w) uint8 matrix of the patterns whose FIRST bit is 0.
    Each row b pairs with its complement ~b (first bit 1), so the rows enumerate the
    2^{w-1} complementary pairs of a weight-w operator exactly once."""
    if w not in _BITS:
        B = np.zeros((1 << (w - 1), w), np.uint8)
        for i in range(1 << (w - 1)):
            for j in range(w - 1):
                B[i, j + 1] = (i >> j) & 1
        _BITS[w] = B
    return _BITS[w]


def group_by_weight(ops):
    g = {}
    for c in ops:
        g.setdefault(len(c), []).append(list(c))
    return {w: np.array(v, int) for w, v in sorted(g.items())}


def exact_class_mass(groups, x):
    """sum_l C_l  for one class, given the class's presented per-qubit marginals x."""
    tot = 0.0
    for w, arr in groups.items():
        xs = np.clip(x[arr], 1e-300, 1 - 1e-12)          # (m, w)
        B = half_patterns(w).astype(np.float64)
        lx, l1 = np.log(xs), np.log1p(-xs)
        lp = lx @ B.T + l1 @ (1.0 - B).T                  # (m, 2^{w-1}) log P[e]
        lq = lx @ (1.0 - B).T + l1 @ B.T                  # log P[e ^ l]
        tot += float(np.exp(np.minimum(lp, lq)).sum())
    return tot


def exact_C(gX, gZ, P3, F):
    rX, rZ = present_s3(P3, F)
    return max(exact_class_mass(gX, rX), exact_class_mass(gZ, rZ))


def surrogate_C_gamma(groups, x):
    """The surrogate's class sum, computed the same operator-by-operator way as the
    exact one (used ONLY for the like-for-like per-weight timing; the deployed path
    is build_U6's single dense product)."""
    tot = 0.0
    for w, arr in groups.items():
        tot += float(gamma(np.clip(x[arr], 0, 0.5)).prod(1).sum())
    return tot


# ---------------------------------------------------------------- rank statistics
def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra, rb = _rank(a), _rank(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _rank(v):
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v), float)
    r[o] = np.arange(len(v), dtype=float)
    # average ties
    s = v[o]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[o[i:j + 1]] = np.arange(i, j + 1).mean()
        i = j + 1
    return r


def kendall(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a); c = d = 0
    for i in range(n):
        da = a[i + 1:] - a[i]; db = b[i + 1:] - b[i]
        s = np.sign(da) * np.sign(db)
        c += int((s > 0).sum()); d += int((s < 0).sum())
    return float((c - d) / max(c + d, 1))


# ---------------------------------------------------------------- frame ensemble
def frame_set(C, n, P3, U6, nrand, seed):
    rng = np.random.default_rng(seed)
    F = {}
    F["css"] = np.zeros(n, int)
    try:
        F["tiurev"] = np.asarray(m_tiurev6(C, P3[:, 0], P3[:, 1], P3[:, 2]), int)
    except Exception as e:                                    # color/BB have no template
        print("   (tiurev frame unavailable: %s)" % e)
    Ub = lambda S: U6(np.where(np.asarray(S, bool), 1, 0))     # binary subspace of S3
    warm = np.array(cem_pool(Ub, n, iters=15, M=200, seed=5)[0][1], int)
    F["cem_binary"] = np.where(warm.astype(bool), 1, 0)
    pool = cem6_pool(U6, n, iters=50, M=500, seed=5, warm=F["cem_binary"], topk=48)
    for i, f in enumerate(pool):
        F["cem6_%02d" % i] = np.asarray(f, int)
    for i in range(nrand):
        F["rand_%03d" % i] = rng.integers(0, 6, n)
    for i in range(nrand // 2):
        F["randbin_%03d" % i] = rng.integers(0, 2, n)
    return F


# ---------------------------------------------------------------- one cell
def run_cell(spec, noise, fseed, p, nrand, do_scaling=True):
    C, _, _ = get_code(spec)
    n = C["n"]
    LX, LZ = mechs(spec, C)
    gX, gZ = group_by_weight(LX), group_by_weight(LZ)
    pX, pY, pZ = field3x(noise, n, fseed, p)
    s = (pX + pY + pZ).mean(); sc = p / max(s, 1e-15)
    pX, pY, pZ = pX * sc, pY * sc, pZ * sc
    P3 = np.stack([pX, pY, pZ], 1)
    U6 = build_U6(LX, LZ, n, pX, pY, pZ)

    F = frame_set(C, n, P3, U6, nrand, seed=1000 + fseed)
    names = list(F)
    Farr = np.array([F[k] for k in names], int)

    t0 = time.perf_counter(); Uv = np.asarray(U6(Farr), float); tU = time.perf_counter() - t0
    t0 = time.perf_counter()
    Cv = np.array([exact_C(gX, gZ, P3, f) for f in Farr], float)
    tC = time.perf_counter() - t0

    ok = np.isfinite(Uv) & np.isfinite(Cv) & (Cv > 0)
    Uv2, Cv2 = Uv[ok], Cv[ok]
    nm = [names[i] for i in np.nonzero(ok)[0]]
    iU, iC = int(np.argmin(Uv2)), int(np.argmin(Cv2))

    def topk_overlap(k):
        a = set(np.argsort(Uv2)[:k].tolist()); b = set(np.argsort(Cv2)[:k].tolist())
        return len(a & b) / k

    # The exact objective is heavily DEGENERATE (many frames share the identical
    # minimum), so a plain top-k index overlap under-reports agreement: it counts a
    # tie-mate as a miss. Report both, and the tie multiplicity that explains it.
    cmin = Cv2.min()
    tie = np.isclose(Cv2, cmin, rtol=1e-9, atol=0.0)

    def topk_overlap_tieaware(k):
        idx = np.argsort(Uv2)[:k]
        thr = np.sort(Cv2)[k - 1]
        return float(np.mean(Cv2[idx] <= thr * (1 + 1e-9)))

    # cost of the two pricings, per frame, and the LUT/working-set memory
    IXb = len(LX) * n * 8; IZb = len(LZ) * n * 8
    exact_ws = max(max((len(a) * (1 << (w - 1)) * 8 for w, a in g.items()), default=0)
                   for g in (gX, gZ))

    res = dict(
        spec=spec, noise=noise, fseed=fseed, p=p, n=n,
        n_mech_X=len(LX), n_mech_Z=len(LZ),
        weights_X={str(w): int(len(a)) for w, a in gX.items()},
        weights_Z={str(w): int(len(a)) for w, a in gZ.items()},
        n_frames=int(ok.sum()),
        t_surrogate_per_frame=tU / max(len(Farr), 1),
        t_exact_per_frame=tC / max(len(Farr), 1),
        speedup=(tC / max(tU, 1e-12)),
        lut_bytes=IXb + IZb, exact_workingset_bytes=int(exact_ws),
        rho_U_C=spearman(Uv2, Cv2), tau_U_C=kendall(Uv2, Cv2),
        top1_overlap=topk_overlap(1), top5_overlap=topk_overlap(5),
        top10_overlap=topk_overlap(10),
        top1_overlap_tieaware=topk_overlap_tieaware(1),
        top5_overlap_tieaware=topk_overlap_tieaware(5),
        top10_overlap_tieaware=topk_overlap_tieaware(10),
        n_C_optimal_ties=int(tie.sum()),
        argmin_U=nm[iU], argmin_C=nm[iC],
        C_at_argminU=float(Cv2[iU]), C_at_argminC=float(Cv2[iC]),
        exact_regret_pct=float(100.0 * (Cv2[iU] / Cv2[iC] - 1.0)),
        U_at_argminU=float(Uv2[iU]), U_at_argminC=float(Uv2[iC]),
        U_min=float(Uv2.min()), U_max=float(Uv2.max()),
        C_min=float(Cv2.min()), C_max=float(Cv2.max()),
        frames={k: [int(x) for x in F[k]] for k in ("css", "cem_binary")
                if k in F},
        frame_argminU=[int(x) for x in Farr[np.nonzero(ok)[0][iU]]],
        frame_argminC=[int(x) for x in Farr[np.nonzero(ok)[0][iC]]],
        per_frame={k: dict(U=float(u), C=float(c))
                   for k, u, c in zip(nm, Uv2, Cv2)},
    )

    # ---- per-weight cost scaling: exact 2^{w-1} pairs vs surrogate w products
    if do_scaling:
        sc_rows = []
        rXv, _ = present_s3(P3, Farr[0])
        for w, arr in gX.items():
            if len(arr) == 0:
                continue
            rep = max(1, min(20, int(2e6 // (len(arr) * (1 << (w - 1)) + 1))))
            t0 = time.perf_counter()
            for _ in range(rep):
                exact_class_mass({w: arr}, rXv)
            te = (time.perf_counter() - t0) / rep / len(arr)
            rep2 = max(1, min(500, int(2e6 // (len(arr) * w + 1))))
            t0 = time.perf_counter()
            for _ in range(rep2):
                surrogate_C_gamma({w: arr}, rXv)
            ts = (time.perf_counter() - t0) / rep2 / len(arr)
            sc_rows.append(dict(w=int(w), n_ops=int(len(arr)), pairs=1 << (w - 1),
                                t_exact_per_op=te, t_gamma_per_op=ts,
                                ratio=te / max(ts, 1e-15)))
        res["weight_scaling"] = sc_rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="surf2d:3,surf2d:5,color2d:3,color2d:5,BB18")
    ap.add_argument("--fields", default="berlin_star,willow_star,xyz:10")
    ap.add_argument("--fseeds", default="0,1")
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--nrand", type=int, default=200)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    rows = []
    for spec in a.specs.split(","):
        for noise in a.fields.split(","):
            for fs in [int(x) for x in a.fseeds.split(",")]:
                t0 = time.time()
                try:
                    r = run_cell(spec, noise, fs, a.p, a.nrand,
                                 do_scaling=(noise == a.fields.split(",")[0] and fs == 0))
                except Exception as e:
                    print("FAIL %s|%s|fs%d: %r" % (spec, noise, fs, e), flush=True)
                    continue
                r["wall_s"] = time.time() - t0
                rows.append(r)
                print("M1 %s|%s|fs%d: rho(U,C)=%.3f tau=%.3f top1=%.0f%% "
                      "regret=%.2f%% speedup=%.0fx (%.0fs)"
                      % (spec, noise, fs, r["rho_U_C"], r["tau_U_C"],
                         100 * r["top1_overlap"], r["exact_regret_pct"],
                         r["speedup"], r["wall_s"]), flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("\nwrote %s (%d cells)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
