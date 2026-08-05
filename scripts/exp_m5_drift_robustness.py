r"""M5 -- does the frame survive an imperfect or stale calibration map?

Everywhere else in the paper the frame is selected on the same map it is graded on, which
is an oracle-calibration assumption. Here the two are separated:

    compile on   \hat P   (what the compiler is told)
    grade on     P        (what the device actually is)

Two perturbation models, both already in chameleon.fields:
  miscalibration  every per-qubit axis rate multiplied by an independent lognormal factor
                  of width sigma  ->  noise spec "drift:<sigma>:<dseed>:<base>"
  stale map       compile on field seed s, grade on field seed s' of the same device,
                  i.e. a different draw of the same calibration family

Reported per cell: the surrogate the deployed frame achieves ON THE TRUE MAP, against
(a) the frame the true map would have selected -- the regret from being misinformed, and
(b) the undeformed CSS frame -- whether the misinformed frame still beats doing nothing,
which is the property that actually matters for deployment.

Outputs results/typeA_investigation/m5_drift_robustness.json
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
from chameleon.fields import field3x
from chameleon.surrogate import build_U6
from chameleon.search import cem_pool, cem6

OUT = rpath(_CFG.study_dir + "/m5_drift_robustness.json")


def norm(pX, pY, pZ, p):
    s = (pX + pY + pZ).mean(); sc = p / max(s, 1e-15)
    return pX * sc, pY * sc, pZ * sc


def pick(U, n, seed=5):
    Ub = lambda S: U(np.where(np.asarray(S, bool), 1, 0))
    warm = np.where(np.asarray(cem_pool(Ub, n, iters=15, M=200, seed=seed)[0][1], bool), 1, 0)
    return np.asarray(cem6(U, n, iters=50, M=500, seed=seed, warm=warm), int)


def run_cell(spec, base, fseed, p, sigmas, dseeds, stale_seeds):
    C, _, _ = get_code(spec)
    n = C["n"]
    LX, LZ = mechs(spec, C)

    # the TRUE map and the frame it would have selected
    tX, tY, tZ = norm(*field3x(base, n, fseed, p), p)
    Utrue = build_U6(LX, LZ, n, tX, tY, tZ)
    F_oracle = pick(Utrue, n)
    u_oracle = float(Utrue(F_oracle[None])[0])
    u_css = float(Utrue(np.zeros(n, int)[None])[0])

    row = dict(spec=spec, base=base, fseed=fseed, p=p, n=n,
               U_oracle=u_oracle, U_css=u_css,
               css_headroom_pct=100.0 * (u_css / u_oracle - 1.0), cases=[])

    def evaluate(tag, cX, cY, cZ):
        """Select on the compiler's map, grade the selected frame on the true map."""
        Uc = build_U6(LX, LZ, n, cX, cY, cZ)
        F = pick(Uc, n)
        u_on_true = float(Utrue(F[None])[0])
        row["cases"].append(dict(
            case=tag,
            U_on_true=u_on_true,
            regret_vs_oracle_pct=100.0 * (u_on_true / u_oracle - 1.0),
            gain_vs_css_pct=100.0 * (1.0 - u_on_true / u_css),
            still_beats_css=bool(u_on_true < u_css),
            hamming_to_oracle=int((F != F_oracle).sum())))
        return row["cases"][-1]

    # --- miscalibration: multiplicative lognormal noise on every axis rate
    for sig in sigmas:
        for ds in dseeds:
            spec_noise = "drift:%g:%d:%s" % (sig, ds, base)
            cX, cY, cZ = norm(*field3x(spec_noise, n, fseed, p), p)
            evaluate("miscal_sigma%g_d%d" % (sig, ds), cX, cY, cZ)

    # --- stale map: a different draw of the same calibration family
    for s2 in stale_seeds:
        if s2 == fseed:
            continue
        cX, cY, cZ = norm(*field3x(base, n, s2, p), p)
        evaluate("stale_fs%d" % s2, cX, cY, cZ)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="surf2d:5,color2d:5,BB18,surf2d:7,BB72")
    ap.add_argument("--fields", default="willow_star,berlin_star,xyz:10")
    ap.add_argument("--fseeds", default="0,1")
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--sigmas", default="0.05,0.1,0.2,0.4")
    ap.add_argument("--dseeds", default="1,2,3")
    ap.add_argument("--stale", default="20,21,22")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    sig = [float(x) for x in a.sigmas.split(",")]
    ds = [int(x) for x in a.dseeds.split(",")]
    st = [int(x) for x in a.stale.split(",")]
    rows = []
    for spec in a.specs.split(","):
        for base in a.fields.split(","):
            for fs in [int(x) for x in a.fseeds.split(",")]:
                t0 = time.time()
                try:
                    r = run_cell(spec, base, fs, a.p, sig, ds, st)
                except Exception as e:
                    print("FAIL %s|%s|fs%d: %r" % (spec, base, fs, e), flush=True); continue
                r["wall_s"] = time.time() - t0
                rows.append(r)
                mis = [c for c in r["cases"] if c["case"].startswith("miscal")]
                sta = [c for c in r["cases"] if c["case"].startswith("stale")]
                print("M5 %-9s %-12s fs%d | CSS headroom %+6.1f%% | miscal regret "
                      "%.2f-%.2f%% | stale regret %.2f-%.2f%% | beats CSS %d/%d (%.0fs)"
                      % (spec, base, fs, r["css_headroom_pct"],
                         min(c["regret_vs_oracle_pct"] for c in mis),
                         max(c["regret_vs_oracle_pct"] for c in mis),
                         min(c["regret_vs_oracle_pct"] for c in sta),
                         max(c["regret_vs_oracle_pct"] for c in sta),
                         sum(c["still_beats_css"] for c in r["cases"]), len(r["cases"]),
                         r["wall_s"]), flush=True)
                json.dump(rows, open(a.out, "w"), indent=1)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("\nwrote %s (%d cells)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
