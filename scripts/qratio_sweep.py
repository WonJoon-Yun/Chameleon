"""Measurement-error-ratio robustness sweep : gain over the best
fixed baseline vs q_ratio = p_m/p_d in {0, 0.1, 0.5} (1.0 = the main protocol matrix),
representative codes {surf2d:5, color2d:5, BB72} x 4 fields x 3 map draws, at the
protocol anchor p and sweep-tier budgets.
Usage: PROCS=20 python3 scripts/qratio_sweep.py
-> results/protocol_v1/qratio_sweep.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

from chameleon.config import BASELINES
import os, sys, dataclasses
from multiprocessing import Pool

from chameleon.config import ProtocolConfig
from chameleon.core import Cell
from chameleon._root import rpath, atomic_json_dump, acquire_lock
from chameleon.records import resume_records

CODES = ("surf2d:5", "color2d:5", "BB72")
RATIOS = (0.0, 0.1, 0.5)
DRAWS = (0, 1, 2)


def main():
    base = ProtocolConfig.default()
    out = rpath(base.out_dir, "qratio_sweep.json")
    acquire_lock(out)
    pool = Pool(int(os.environ.get("PROCS", "20")), maxtasksperchild=100)
    results = resume_records(out)
    done = {(r["spec"], r["noise"], r["fseed"], r["q_ratio"]) for r in results}
    jobs = [(spec, f, s, ratio) for ratio in RATIOS for spec in CODES
            for f in base.fields for s in DRAWS]
    jobs = [j for j in jobs if j not in done]
    print("qratio cells to run:", len(jobs), flush=True)
    for spec, noise, fseed, ratio in jobs:
        cfg = dataclasses.replace(base, q_ratio=ratio)
        rec = Cell(cfg, spec, noise, cfg.p_anchor, fseed, "eta").run(pool)
        rec["tier"] = "qratio"
        m = rec["masks"]
        bb = min(m[k]["ler"] for k in BASELINES)
        print("[%s %-11s s%d q/p=%.1f] CSS %.3e bb %.3e Cham %.3e xBB %s ev=%d" % (
            spec, noise, fseed, ratio, m["CSS"]["ler"], bb, m["Cham"]["ler"],
            ("%.2f" % rec["gain_bb"]) if rec["gain_bb"] else "n/a", m["Cham"]["ev"]), flush=True)
        results.append(rec); atomic_json_dump(results, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    main()
