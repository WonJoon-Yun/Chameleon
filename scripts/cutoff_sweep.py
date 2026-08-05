"""Mechanism-cutoff sensitivity under protocol v1.6 (replaces the earlier
code-capacity P=0.05 study behind fig_cutoff).

For each BB code and cutoff c in {70, 80, 90, 95, 100}% of the enumerated
gamma-product mass (computed on the map, CSS frame), select the blind top-1
frame using only the mechanisms inside the cutoff, then measure it and the
fixed baselines under the full v1.6 protocol (xyz:10, p=0.005, q=p, T=d).
  PROCS=30 python3 scripts/cutoff_sweep.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

from chameleon.config import BASELINES
import os, sys, hashlib
import numpy as np
from multiprocessing import Pool
from chameleon.config import ProtocolConfig
from chameleon.core import Code, NoiseField, BlindS3Selector, PhenoEvaluator, baseline_frames
from chameleon._root import rpath, atomic_json_dump, acquire_lock
from chameleon.records import resume_records

CODES = ("BB72", "BB18", "BB30")   # user pivot 2026-07-13
CUTS = (0.70, 0.80, 0.90, 0.95, 1.00)
DRAWS = (0, 1, 2)
NOISE = "xyz:10"


def truncate(code, field, cut):
    """Keep the lightest-gamma-mass mechanisms holding `cut` of the total mass
    (per axis, mass computed at the CSS frame on this map)."""
    if cut >= 1.0 - 1e-12:
        return code.mechs
    P3 = np.stack([field.pX, field.pY, field.pZ], 1)
    out = []
    for Ls, (a, b) in zip(code.mechs, ((0, 1), (2, 1))):
        r = P3[:, a] + P3[:, b]
        g = 2 * np.sqrt(np.clip(r * (1 - r), 0, None))
        w = np.array([np.prod(g[list(m)]) for m in Ls])
        order = np.argsort(-w)
        keep = order[: int(np.searchsorted(np.cumsum(w[order]), cut * w.sum()) + 1)]
        out.append([list(Ls[i]) for i in sorted(keep)])
    return tuple(out)


def main():
    cfg = ProtocolConfig()
    cfg_id = "%s:%s" % (cfg.name, hashlib.md5(cfg.to_json().encode()).hexdigest()[:8])
    out = rpath(cfg.out_dir, "cutoff_sweep.json")
    acquire_lock(out)
    results = resume_records(out)
    prev_ids = {r.get("cfg_id") for r in results if r.get("cfg_id")}
    if prev_ids and prev_ids != {cfg_id}:
        raise SystemExit("refusing to mix budgets: %s vs %s" % (prev_ids, cfg_id))
    done = {(r["spec"], r["fseed"], r["cut"]) for r in results}

    sel = BlindS3Selector(cfg.selector)
    procs = int(os.environ.get("PROCS", "30"))
    pool = Pool(procs, maxtasksperchild=100)
    for spec in CODES:
        code = Code(spec)
        full = code.mechs
        for s in DRAWS:
            field = NoiseField(NOISE, cfg.p_anchor, s, code)
            ev = PhenoEvaluator(cfg, pool)
            bb_ler = None
            for cut in CUTS:
                if (spec, s, cut) in done:
                    continue
                code._mechs = truncate(code, field, cut)
                frame = sel.select(code, field)
                code._mechs = full
                if bb_ler is None:
                    bl = {k: ev.worst_axis(code, field, f, "eta").as_dict()
                          for k, f in baseline_frames(code, field).items()}
                    bb_ler = bl
                m = ev.worst_axis(code, field, frame, "eta").as_dict()
                nm = [len(x) for x in truncate(code, field, cut)]
                bb = min(v["ler"] for k, v in bb_ler.items()
                         if k in BASELINES)
                rec = dict(spec=spec, fseed=s, cut=cut, nmech=nm, cham=m,
                           baselines=bb_ler, cfg_id=cfg_id,
                           gain_bb=(bb / m["ler"]) if m["ler"] > 0 and not m["unresolved"] else None)
                print("[%s s%d cut=%.2f nm=%s] Cham %.3e bb %.3e gain %s%s" % (
                    spec, s, cut, nm, m["ler"], bb,
                    ("%.2f" % rec["gain_bb"]) if rec["gain_bb"] else "n/a",
                    " UNRESOLVED" if m["unresolved"] else ""), flush=True)
                results.append(rec)
                atomic_json_dump(results, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    main()
