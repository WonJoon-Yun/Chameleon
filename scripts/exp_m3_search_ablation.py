"""M3 -- why CEM?  Equal-budget search ablation + exhaustive optimality gap.

Every generator gets the SAME number of surrogate evaluations (the deployed
warm+S3 schedule's budget: 15*200 binary + 3*50*500 S3 = 78,000), so the comparison
isolates the SEARCH, not the compute.  Generators:

  random        uniform S3 sampling                        (the current paper's only baseline)
  local_rule    per-qubit dominant-axis rule (Tiurev)      1 eval, the no-search reference
  greedy_cd     coordinate descent from CSS
  ms_greedy     multi-start coordinate descent
  anneal        simulated annealing, single-qubit moves
  binary_cem    cem_pool over {X,Z}^n only (no S3 stage)
  cem6_cold     cem6 with three COLD restarts (no binary warm start)
  cem6_warm     DEPLOYED: binary warm start + cem6                (chameleon.core path)

On surf2d:3 the whole 6^9 = 10,077,696 frame space is enumerated exactly, giving the
true optimum U* and therefore the optimality gap (U_alg - U*) / U* for every generator.

Outputs results/typeA_investigation/m3_search_ablation.json
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
from chameleon.vendor.multicode_deform import m_tiurev6

OUT = rpath(_CFG.study_dir + "/m3_search_ablation.json")
BUDGET = 78_000          # = 15*200 (binary warm) + 3*50*500 (S3), the deployed schedule


class Counter:
    """Wraps the surrogate so every generator is charged for what it actually scores,
    and records the best-so-far trajectory against the evaluation count."""

    def __init__(self, U, budget):
        self.U, self.budget, self.n = U, budget, 0
        self.best = np.inf; self.best_f = None; self.curve = []

    def __call__(self, F):
        F = np.atleast_2d(np.asarray(F, int))
        if self.n >= self.budget:
            raise StopIteration
        if self.n + len(F) > self.budget:
            F = F[: self.budget - self.n]
        u = np.asarray(self.U(F), float)
        self.n += len(F)
        i = int(np.argmin(u))
        if u[i] < self.best:
            self.best = float(u[i]); self.best_f = F[i].copy()
            self.curve.append((self.n, self.best))
        return u


def g_random(cnt, n, rng):
    while cnt.n < cnt.budget:
        cnt(rng.integers(0, 6, (min(2000, cnt.budget - cnt.n), n)))


def g_greedy_cd(cnt, n, rng, start=None):
    """Coordinate descent. Every score it uses is charged to the budget -- the current
    incumbent's value is carried, never re-scored off the meter."""
    F = np.zeros(n, int) if start is None else np.asarray(start, int).copy()
    cur = float(cnt(F[None])[0])
    improved = True
    while improved and cnt.n < cnt.budget:
        improved = False
        for q in range(n):
            if cnt.n >= cnt.budget:
                return F
            cand = np.repeat(F[None], 6, 0); cand[:, q] = np.arange(6)
            u = cnt(cand)
            j = int(np.argmin(u))
            if u[j] < cur - 1e-15:
                F = cand[j].copy(); cur = float(u[j]); improved = True
    return F


def g_ms_greedy(cnt, n, rng):
    try:
        g_greedy_cd(cnt, n, rng, np.zeros(n, int))
        while cnt.n < cnt.budget:
            g_greedy_cd(cnt, n, rng, rng.integers(0, 6, n))
    except StopIteration:
        pass  # the evaluation counter raises when the budget is spent; stopping there is the point


def g_anneal(cnt, n, rng):
    F = rng.integers(0, 6, n); cur = float(cnt(F[None])[0])
    T0, T1 = max(cur * 0.05, 1e-30), max(cur * 1e-4, 1e-32)
    while cnt.n < cnt.budget:
        frac = cnt.n / cnt.budget
        T = T0 * (T1 / T0) ** frac
        G = np.repeat(F[None], 64, 0)
        qs = rng.integers(0, n, 64); vs = rng.integers(0, 6, 64)
        G[np.arange(64), qs] = vs
        u = cnt(G)
        for k in range(len(u)):
            d = u[k] - cur
            if d < 0 or rng.random() < np.exp(-d / max(T, 1e-300)):
                F = G[k].copy(); cur = float(u[k])


def g_binary_cem(cnt, n, rng):
    Ub = lambda S: cnt(np.where(np.asarray(S, bool), 1, 0))
    it = 0
    while cnt.n < cnt.budget:
        try:
            cem_pool(Ub, n, iters=15, M=200, seed=int(rng.integers(1 << 30)))
        except StopIteration:
            return
        it += 1
        if it > 10_000:
            return


def g_cem6_cold(cnt, n, rng):
    try:
        cem6(cnt, n, iters=50, M=500, seed=int(rng.integers(1 << 30)),
             warm=None, restarts=[None, None, None])
    except StopIteration:
        pass  # the evaluation counter raises when the budget is spent; stopping there is the point


def g_cem6_warm(cnt, n, rng):
    """The DEPLOYED schedule, verbatim in structure: binary CEM warm start (15x200)
    then the S3 CEM with restarts [cold, cold, warm]."""
    try:
        Ub = lambda S: cnt(np.where(np.asarray(S, bool), 1, 0))
        pool = cem_pool(Ub, n, iters=15, M=200, seed=5)
        warm = np.where(np.asarray(pool[0][1], bool), 1, 0)
        cem6(cnt, n, iters=50, M=500, seed=5, warm=warm)
    except StopIteration:
        pass  # the evaluation counter raises when the budget is spent; stopping there is the point


GENS = {"random": g_random, "greedy_cd": lambda c, n, r: g_greedy_cd(c, n, r),
        "ms_greedy": g_ms_greedy, "anneal": g_anneal,
        "binary_cem": g_binary_cem, "cem6_cold": g_cem6_cold, "cem6_warm": g_cem6_warm}


def exhaustive_min(U, n, chunk=200_000):
    """Exact minimum over the full 6^n space (only tractable for n<=9)."""
    tot = 6 ** n
    best = np.inf; bf = None
    digits = 6 ** np.arange(n)
    for s in range(0, tot, chunk):
        idx = np.arange(s, min(s + chunk, tot))
        F = (idx[:, None] // digits) % 6
        u = np.asarray(U(F), float)
        i = int(np.argmin(u))
        if u[i] < best:
            best = float(u[i]); bf = F[i].copy()
    return best, bf, tot


def run_cell(spec, noise, fseed, p, seeds, budget, do_exhaustive):
    C, _, _ = get_code(spec)
    n = C["n"]
    LX, LZ = mechs(spec, C)
    pX, pY, pZ = field3x(noise, n, fseed, p)
    s = (pX + pY + pZ).mean(); sc = p / max(s, 1e-15)
    pX, pY, pZ = pX * sc, pY * sc, pZ * sc
    U = build_U6(LX, LZ, n, pX, pY, pZ)

    row = dict(spec=spec, noise=noise, fseed=fseed, p=p, n=n, budget=budget,
               n_mech_X=len(LX), n_mech_Z=len(LZ), methods={})

    # --- no-search reference: the per-qubit local rule
    try:
        FT = np.asarray(m_tiurev6(C, pX, pY, pZ), int)
        row["methods"]["local_rule"] = dict(
            best_U=[float(U(FT[None])[0])], evals=[1], wall_s=[0.0],
            frame=[int(x) for x in FT])
    except Exception as e:
        row["local_rule_error"] = repr(e)
    row["methods"]["css"] = dict(best_U=[float(U(np.zeros(n, int)[None])[0])],
                                 evals=[1], wall_s=[0.0])

    for name, fn in GENS.items():
        bs, ev, wl, frames = [], [], [], []
        curves = []
        for sd in seeds:
            rng = np.random.default_rng(sd)
            cnt = Counter(U, budget)
            t0 = time.perf_counter()
            try:
                fn(cnt, n, rng)
            except StopIteration:
                pass  # the evaluation counter raises when the budget is spent; stopping there is the point
            wl.append(time.perf_counter() - t0)
            bs.append(cnt.best); ev.append(cnt.n)
            frames.append([int(x) for x in cnt.best_f] if cnt.best_f is not None else None)
            curves.append(cnt.curve)
        row["methods"][name] = dict(best_U=bs, evals=ev, wall_s=wl, frame=frames[0],
                                    curve=curves[0])
        print("   %-11s bestU min/max/mean = %.6e / %.6e / %.6e  (%d evals, %.1fs)"
              % (name, min(bs), max(bs), float(np.mean(bs)), int(np.mean(ev)),
                 float(np.mean(wl))), flush=True)

    if do_exhaustive and n <= 9:
        t0 = time.perf_counter()
        ustar, fstar, tot = exhaustive_min(U, n)
        row["exhaustive"] = dict(U_star=ustar, frame=[int(x) for x in fstar],
                                 space=tot, wall_s=time.perf_counter() - t0)
        for name, d in row["methods"].items():
            d["gap_pct"] = [100.0 * (b / ustar - 1.0) for b in d["best_U"]]
        print("   EXHAUSTIVE 6^%d=%d  U*=%.6e (%.0fs)  gaps: %s"
              % (n, tot, ustar, row["exhaustive"]["wall_s"],
                 {k: round(float(np.mean(v["gap_pct"])), 3)
                  for k, v in row["methods"].items()}), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="surf2d:3,surf2d:5,color2d:5,BB18,BB72")
    ap.add_argument("--fields", default="berlin_star,willow_star,xyz:10")
    ap.add_argument("--fseeds", default="0,1")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    rows = []
    for spec in a.specs.split(","):
        for noise in a.fields.split(","):
            for fs in [int(x) for x in a.fseeds.split(",")]:
                print("M3 %s|%s|fs%d" % (spec, noise, fs), flush=True)
                t0 = time.time()
                try:
                    r = run_cell(spec, noise, fs, a.p, seeds, a.budget,
                                 do_exhaustive=spec.startswith("surf2d:3"))
                except Exception as e:
                    print("FAIL %s|%s|fs%d: %r" % (spec, noise, fs, e), flush=True)
                    continue
                r["wall_s"] = time.time() - t0
                rows.append(r)
                json.dump(rows, open(a.out, "w"), indent=1)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("\nwrote %s (%d cells)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
