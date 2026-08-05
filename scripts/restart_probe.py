"""Verify the SS5 hyperparameter claim: independent CEM restarts reach the same
basin. Runs 20 independent cold restarts (now expressible via the plumbed
restarts= parameter) on representative cells and reports the per-restart best-U
trajectory: iteration at which each restart is within 1% of its final value, and
the spread of final U across restarts.  Selection only -- no decoding.
  python3 scripts/restart_probe.py
-> results/protocol_v1/restart_probe.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads(8)          # before numpy: BLAS pools are sized at import

from chameleon.config import ProtocolConfig
_CFG = ProtocolConfig()
import numpy as np

from chameleon.config import ProtocolConfig
from chameleon.core import Code, NoiseField
import chameleon.surrogate as sur
from chameleon._root import rpath, atomic_json_dump

CELLS = [("surf2d:5", "berlin_star", 0), ("color2d:5", "willow_star", 0), ("BB72", "xyz:10", 0)]
RESTARTS, ITERS, M = 20, 50, 500


def main():
    cfg = ProtocolConfig.default()
    out = []
    for spec, noise, fseed in CELLS:
        code = Code(spec)
        fld = NoiseField(noise, cfg.p_anchor, fseed, code)
        LX, LZ = code.mechs
        U = sur.build_U6(LX, LZ, code.n, fld.pX, fld.pY, fld.pZ)
        finals, hit_iters = [], []
        for r in range(RESTARTS):
            rng = np.random.default_rng(1000 + 97 * r)
            prob = np.full((code.n, 6), 1 / 6.0)
            best = np.inf
            traj = []
            for t in range(ITERS):
                S = np.array([rng.choice(6, size=M, p=prob[q]) for q in range(code.n)]).T
                u = U(S)
                idx = np.argsort(u)[:max(2, M // 10)]
                best = min(best, float(u[idx[0]]))
                traj.append(best)
                emp = np.zeros((code.n, 6))
                for i in idx:
                    emp[np.arange(code.n), S[i]] += 1
                emp /= len(idx)
                prob = 0.3 * prob + 0.7 * emp
                prob = np.clip(prob, 0.01, None)
                prob /= prob.sum(1, keepdims=True)
            finals.append(best)
            hit = next(t for t, v in enumerate(traj) if v <= best * 1.01)
            hit_iters.append(hit + 1)
        rec = dict(spec=spec, noise=noise, fseed=fseed, restarts=RESTARTS,
                   final_spread=float(max(finals) / min(finals)),
                   hit_iter_median=int(np.median(hit_iters)),
                   hit_iter_max=int(max(hit_iters)))
        out.append(rec)
        print("[%s %s] final-U spread %.4f, within-1%%-of-final by iter median %d / max %d" % (
            spec, noise, rec["final_spread"], rec["hit_iter_median"], rec["hit_iter_max"]), flush=True)
        atomic_json_dump(out, rpath(_CFG.out_dir + "/restart_probe.json"))
    print("DONE", flush=True)


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    main()
