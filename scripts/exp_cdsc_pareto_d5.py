"""CDSC + random-U quality-cost Pareto on surf d=5, Willow fs0, p=0.02
(surface d=5 has a 2^25 binary frame space, so the exhaustive sweep is feasible there).

At d=5 the binary space is 2^25 = 33.5M: still exhaustively U-scorable (Phase A)
but no longer exhaustively MEASURABLE, so the CDSC arm follows the real protocol:

  Phase B  validation: ONE pool of 10,000 random binary candidates, each
           decode-validated cheaply (MIN_FAIL=100/axis, +-10%) -- the user's
           "run 10K once, slice the prefixes" design.
  Phase C  slicing: 20 permutation streams over the SAME pool; budget N picks
           the best-validation-LER candidate among the stream's first N.
           Selection keeps the realistic winner's-curse noise; EVALUATION does
           not: every distinct winner is re-measured at MIN_FAIL=3000.
           The random-S3 best-U arm (20 streams x 10^6 draws, prefix argmin-U)
           is measured the same way for the same budgets.

References: CSS, the exhaustive binary-U optimum (Phase A, measured), and the
deployed Chameleon frame for this exact cell from s7_ler_surf5.json.

    PROCS=160 python3 scripts/exp_cdsc_pareto_d5.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

from chameleon.config import ProtocolConfig
_CFG = ProtocolConfig()
import os, sys, json, time
import numpy as np
from multiprocessing import Pool

SPEC, NOISE, FSEED, P = "surf2d:5", "willow_star", 0, 0.02
BUDGETS = [1, 5, 10, 24, 50, 100, 500, 1000, 5000, 10000]
UBUDGETS = [1, 5, 10, 24, 50, 100, 500, 1000, 5000, 10000, 100000, 1000000]
STREAMS, NPOOL = 20, 10000
OUT = _CFG.study_dir + "/cdsc_pareto_surf5.json"

_FIX = {}


def fixture():
    if not _FIX:
        from chameleon.codes import get_code
        from chameleon.mechs import mechs
        from chameleon.fields import field3x
        from chameleon.surrogate import build_U6
        C, _, _ = get_code(SPEC)
        n = C["n"]
        LX, LZ = mechs(SPEC, C)
        pX, pY, pZ = field3x(NOISE, n, FSEED, P)
        s = (pX + pY + pZ).mean()
        sc = P / max(s, 1e-15)
        _FIX.update(n=n, U6=build_U6(LX, LZ, n, pX * sc, pY * sc, pZ * sc))
    return _FIX


def ub_chunk(args):
    lo, hi = args
    fx = fixture()
    n, U6 = fx["n"], fx["U6"]
    idx = np.arange(lo, hi, dtype=np.int64)
    F = np.zeros((len(idx), n), np.int8)
    for q in range(n):
        F[:, q] = (idx >> q) & 1
    u = U6(F.astype(int))
    j = int(np.argmin(u))
    return float(u.min()), int(idx[j])


def us_stream(s):
    fx = fixture()
    n, U6 = fx["n"], fx["U6"]
    rng = np.random.default_rng(3000 + s)
    best_u, best_f, done, picks = np.inf, None, 0, {}
    for N in UBUDGETS:
        draw = rng.integers(0, 6, size=(N - done, n))
        if len(draw):
            u = U6(draw.astype(int))
            j = int(np.argmin(u))
            if float(u[j]) < best_u:
                best_u, best_f = float(u[j]), [int(x) for x in draw[j]]
        done = N
        picks[N] = dict(u=best_u, frame=list(best_f))
    return s, picks


def main():
    t0 = time.time()
    import typeA_pheno as TA
    TA.P = P
    TA.MAX_WAVES = 400
    procs = int(os.environ.get("PROCS", "160"))
    fx = fixture()
    n, U6 = fx["n"], fx["U6"]

    with Pool(procs) as pool:
        # -- Phase A: exhaustive binary U over 2^25 -------------------------------
        total = 2 ** n
        chunks = [(lo, min(lo + 500_000, total)) for lo in range(0, total, 500_000)]
        res = pool.map(ub_chunk, chunks)
        k = int(np.argmin([r[0] for r in res]))
        ub_opt, idx = res[k]
        Fb = [(idx >> q) & 1 for q in range(n)]
        print("binary exhaustive: %d frames, u_opt=%.4e (%.0fs)"
              % (total, ub_opt, time.time() - t0), flush=True)

        # -- random-U S3 streams --------------------------------------------------
        us = dict(pool.map(us_stream, range(STREAMS)))
        print("U-streams done (%.0fs)" % (time.time() - t0), flush=True)

        # -- Phase B: CDSC validation pool ---------------------------------------
        rng = np.random.default_rng(2000)
        cand = rng.integers(0, 2, size=(NPOOL, n))
        vframes = {"v%04d" % i: [int(x) for x in cand[i]] for i in range(NPOOL)}
        TA.MIN_FAIL, TA.WAVE_PER = 100, 1
        vmeas = dict(TA.measure_cell(SPEC, NOISE, FSEED, vframes, pool, 91_000))
        vler = np.array([vmeas["v%04d" % i]["lerA"] for i in range(NPOOL)])
        print("validated %d candidates (%.0fs), val-best %.3e"
              % (NPOOL, time.time() - t0, vler.min()), flush=True)

        # -- Phase C: prefix picks + high-precision re-measurement ---------------
        cpicks = {}
        for s in range(STREAMS):
            prng = np.random.default_rng(1000 + s)
            order = prng.permutation(NPOOL)
            cpicks[s] = {}
            for N in BUDGETS:
                head = order[:N]
                j = int(head[np.argmin(vler[head])])
                cpicks[s][N] = dict(idx=j, val_lerA=float(vler[j]))
        upicks = us
        final = {"CSS": [0] * n, "UoptBin": [int(x) for x in Fb]}
        key_of = {}
        def reg(fr):
            kf = tuple(fr)
            if kf not in key_of:
                key_of[kf] = "m%03d" % len(key_of)
                final[key_of[kf]] = list(kf)
            return key_of[kf]
        for s in range(STREAMS):
            for N in BUDGETS:
                cpicks[s][N]["frame_key"] = reg(vframes["v%04d" % cpicks[s][N]["idx"]])
            for N in UBUDGETS:
                upicks[s][N]["frame_key"] = reg(upicks[s][N]["frame"])
        TA.MIN_FAIL, TA.WAVE_PER = 3000, 16
        print("re-measuring %d distinct frames" % len(final), flush=True)
        meas = dict(TA.measure_cell(SPEC, NOISE, FSEED, final, pool, 92_000))

    out = dict(spec=SPEC, noise=NOISE, fseed=FSEED, p=P, streams=STREAMS,
               budgets=BUDGETS, ubudgets=UBUDGETS, npool=NPOOL,
               ub_opt=ub_opt, u_css=float(U6(np.zeros((1, n), int))[0]),
               cpicks={str(s): {str(N): d for N, d in p.items()} for s, p in cpicks.items()},
               upicks={str(s): {str(N): {k: v for k, v in d.items() if k != "frame"}
                                for N, d in p.items()} for s, p in upicks.items()},
               meas=meas, wall_s=time.time() - t0)
    json.dump(out, open(OUT, "w"))
    print("wrote %s (%.0fs)" % (OUT, out["wall_s"]), flush=True)
    for N in BUDGETS:
        g = [meas[cpicks[s][N]["frame_key"]]["lerA"] for s in range(STREAMS)]
        print("CDSC N=%-6d LER mean %.3e [%.3e, %.3e]" % (N, np.mean(g), min(g), max(g)), flush=True)
    for N in UBUDGETS:
        g = [meas[upicks[s][N]["frame_key"]]["lerA"] for s in range(STREAMS)]
        print("U    N=%-8d LER mean %.3e [%.3e, %.3e]" % (N, np.mean(g), min(g), max(g)), flush=True)
    print("CSS %.3e  UoptBin %.3e" % (meas["CSS"]["lerA"], meas["UoptBin"]["lerA"]), flush=True)


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    main()
