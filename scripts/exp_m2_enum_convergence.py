"""M2 -- is the retained ambiguity set deep enough?

Two questions the paper's stopping criteria (geometric: omitted surrogate tail <1%;
BB: <0.5% new representatives) assert but never demonstrate:

  (A) geometric codes -- if the enumeration goes DEEPER than the deployed w_min+2,
      does the SELECTED FRAME change?  Sweep the absolute weight cutoff
      W = w_min .. w_min+k and, at each depth, record the mechanism count, the
      enumeration time, the LUT memory, the frame the deployed search selects,
      its Hamming distance to the deepest-depth selection, and -- crucially --
      the deepest-depth surrogate value of every depth's pick (the regret of
      having stopped early, measured on the most complete set available).

  (B) BB codes -- the orbit-enriched set is RANDOMISED (randomised Gaussian
      elimination + orbit expansion).  Re-run the enumeration under different
      seeds and measure set overlap (Jaccard), U-ranking stability across the
      independently enumerated sets, and whether the selected frame moves.

Outputs results/typeA_investigation/m2_enum_convergence.json
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
from chameleon.mechs import mechs, logops
from chameleon.fields import field3x
from chameleon.surrogate import build_U6
from chameleon.search import cem_pool, cem6

OUT = rpath(_CFG.study_dir + "/m2_enum_convergence.json")


def deployed_pick(U, n, seed=5):
    """The deployed selection path: binary CEM warm start -> S3 CEM (chameleon.core)."""
    Ub = lambda S: U(np.where(np.asarray(S, bool), 1, 0))
    warm = np.where(np.asarray(cem_pool(Ub, n, iters=15, M=200, seed=seed)[0][1], bool), 1, 0)
    return np.asarray(cem6(U, n, iters=50, M=500, seed=seed, warm=warm), int)


def build_field(spec, n, noise, fseed, p):
    pX, pY, pZ = field3x(noise, n, fseed, p)
    s = (pX + pY + pZ).mean(); sc = p / max(s, 1e-15)
    return pX * sc, pY * sc, pZ * sc


def geometric_sweep(spec, noise, fseed, p, extra_max, scancap, seeds):
    C, _, _ = get_code(spec)
    n = C["n"]
    pX, pY, pZ = build_field(spec, n, noise, fseed, p)

    # deployed reference set (whatever mechs() actually ships) for context
    LXd, LZd = mechs(spec, C)
    depths = []
    wmin = None
    for extra in range(0, extra_max + 1):
        t0 = time.perf_counter()
        try:
            opsX = logops(C["Hz"], C["LzZ"], n, extra=extra, scancap=scancap,
                          cap=200000, strict=True)
            opsZ = logops(C["Hx"], C["LxX"], n, extra=extra, scancap=scancap,
                          cap=200000, strict=True)
        except Exception as e:
            print("     depth extra=%d: ABORT %r" % (extra, e), flush=True)
            break
        te = time.perf_counter() - t0
        LX = [list(c) for _, c in opsX]; LZ = [list(c) for _, c in opsZ]
        if not LX or not LZ:
            continue
        if wmin is None:
            wmin = min(min(len(c) for c in LX), min(len(c) for c in LZ))
        U = build_U6(LX, LZ, n, pX, pY, pZ)
        picks = [deployed_pick(U, n, seed=s) for s in seeds]
        depths.append(dict(extra=extra, wcut=wmin + extra, n_X=len(LX), n_Z=len(LZ),
                           enum_s=te, lut_bytes=(len(LX) + len(LZ)) * n * 8,
                           picks=[[int(x) for x in f] for f in picks],
                           U_self=[float(U(f[None])[0]) for f in picks],
                           _LX=LX, _LZ=LZ))
        print("     extra=%d w<=%d: |A_X|=%d |A_Z|=%d enum=%.1fs U=%.4e"
              % (extra, wmin + extra, len(LX), len(LZ), te, depths[-1]["U_self"][0]),
              flush=True)
    if not depths:
        return None

    # score every depth's pick on the DEEPEST available set -- the early-stop regret
    deep = depths[-1]
    Udeep = build_U6(deep["_LX"], deep["_LZ"], n, pX, pY, pZ)
    ref = np.asarray(deep["picks"][0], int)
    for d in depths:
        d["U_on_deepest"] = [float(Udeep(np.asarray(f, int)[None])[0]) for f in d["picks"]]
        d["hamming_to_deepest"] = [int((np.asarray(f, int) != ref).sum()) for f in d["picks"]]
    best_deep = min(min(d["U_on_deepest"]) for d in depths)
    for d in depths:
        d["regret_pct_on_deepest"] = [100.0 * (u / best_deep - 1.0) for u in d["U_on_deepest"]]
        d.pop("_LX"); d.pop("_LZ")

    return dict(spec=spec, noise=noise, fseed=fseed, p=p, n=n, w_min=wmin,
                deployed_nX=len(LXd), deployed_nZ=len(LZd), depths=depths,
                U_best_on_deepest=best_deep)


def bb_seed_stability(spec, noise, fseed, p, perms_list, seeds):
    """Independent orbit-enriched enumerations -> set overlap + selection stability."""
    from chameleon.vendor.bb_mech_orbit import enrich
    C, _, _ = get_code(spec)
    n = C["n"]
    pX, pY, pZ = build_field(spec, n, noise, fseed, p)
    wh = json.load(open(rpath(_CFG.mech_dir, "wt_histogram.json")))[spec]
    d = {"BB72": 6, "BB90": 10, "BB108": 10, "BB144": 12}.get(spec, 6)
    extra = max(2, max(wh["weights"]) - d) + 2

    runs = []
    for perms, s0 in perms_list:
        t0 = time.perf_counter()
        LXe, LZe = enrich(spec, extra=extra, perms=perms, orbit=True, seed0=s0)
        te = time.perf_counter() - t0
        LX = [list(s) for s in LXe]; LZ = [list(s) for s in LZe]
        U = build_U6(LX, LZ, n, pX, pY, pZ)
        picks = [deployed_pick(U, n, seed=s) for s in seeds]
        runs.append(dict(perms=perms, seed0=s0, enum_s=te, n_X=len(LX), n_Z=len(LZ),
                         setX=set(map(tuple, map(sorted, LX))),
                         setZ=set(map(tuple, map(sorted, LZ))),
                         picks=[[int(x) for x in f] for f in picks],
                         U_self=[float(U(f[None])[0]) for f in picks], U=U))
        print("     perms=%d seed0=%d: |A_X|=%d |A_Z|=%d enum=%.0fs U=%.4e"
              % (perms, s0, len(LX), len(LZ), te, runs[-1]["U_self"][0]), flush=True)

    ref = runs[-1]
    out_runs = []
    for r in runs:
        jx = len(r["setX"] & ref["setX"]) / max(len(r["setX"] | ref["setX"]), 1)
        jz = len(r["setZ"] & ref["setZ"]) / max(len(r["setZ"] | ref["setZ"]), 1)
        # cross-score: this run's picks priced on the reference (largest) set
        u_cross = [float(ref["U"](np.asarray(f, int)[None])[0]) for f in r["picks"]]
        out_runs.append(dict(perms=r["perms"], seed0=r["seed0"], enum_s=r["enum_s"], n_X=r["n_X"],
                             n_Z=r["n_Z"], jaccard_X=jx, jaccard_Z=jz,
                             picks=r["picks"], U_self=r["U_self"], U_on_ref=u_cross,
                             hamming_to_ref=[int((np.asarray(f, int) !=
                                                  np.asarray(ref["picks"][0], int)).sum())
                                             for f in r["picks"]]))
    best = min(min(r["U_on_ref"]) for r in out_runs)
    for r in out_runs:
        r["regret_pct_on_ref"] = [100.0 * (u / best - 1.0) for u in r["U_on_ref"]]
    return dict(spec=spec, noise=noise, fseed=fseed, p=p, n=n, extra=extra,
                runs=out_runs, U_best_on_ref=best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geo", default="surf2d:3,color2d:3,surf2d:5,color2d:5")
    ap.add_argument("--bb", default="BB72")
    ap.add_argument("--fields", default="berlin_star,willow_star,xyz:10")
    ap.add_argument("--fseeds", default="0")
    ap.add_argument("--seeds", default="5,11,23")
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--extra-max", type=int, default=4)
    ap.add_argument("--scancap", type=int, default=200_000_000)
    ap.add_argument("--perms", default="800:1,800:2,800:3,400:1,200:1")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    res = {"geometric": [], "bb": []}

    for spec in [s for s in a.geo.split(",") if s]:
        for noise in a.fields.split(","):
            for fs in [int(x) for x in a.fseeds.split(",")]:
                print("M2-geo %s|%s|fs%d" % (spec, noise, fs), flush=True)
                try:
                    r = geometric_sweep(spec, noise, fs, a.p, a.extra_max, a.scancap, seeds)
                except Exception as e:
                    print("  FAIL %r" % (e,), flush=True); continue
                if r:
                    res["geometric"].append(r)
                    json.dump(res, open(a.out, "w"), indent=1)

    for spec in [s for s in a.bb.split(",") if s]:
        for noise in a.fields.split(","):
            for fs in [int(x) for x in a.fseeds.split(",")]:
                print("M2-bb %s|%s|fs%d" % (spec, noise, fs), flush=True)
                try:
                    r = bb_seed_stability(spec, noise, fs, a.p,
                                          [tuple(int(y) for y in c.split(":")) for c in a.perms.split(",")], seeds)
                except Exception as e:
                    print("  FAIL %r" % (e,), flush=True); continue
                res["bb"].append(r)
                json.dump(res, open(a.out, "w"), indent=1)

    json.dump(res, open(a.out, "w"), indent=1)
    print("\nwrote %s (geo %d, bb %d)" % (a.out, len(res["geometric"]), len(res["bb"])))


if __name__ == "__main__":
    main()
