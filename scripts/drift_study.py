"""Calibration-drift robustness (paper SS7): the frame is selected on the NOMINAL
map and measured on a DRIFTED map (every per-qubit axis rate multiplied by an
independent lognormal factor of width sigma), against baselines measured on the
same drifted map. Retention = gain(drifted) relative to gain(nominal).
Current protocol defaults, eta-tier budget.
  PROCS=15 python3 scripts/drift_study.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

from chameleon.config import BASELINES
import os, sys, hashlib
from multiprocessing import Pool
from chameleon.config import ProtocolConfig
from chameleon.core import Code, NoiseField, BlindS3Selector, PhenoEvaluator, baseline_frames
from chameleon._root import rpath, atomic_json_dump, acquire_lock
from chameleon.records import resume_records

CODES = ("surf2d:5", "color2d:5")
BASES = ("willow_star", "xyz:10")
SIGMAS = (0.0, 0.1, 0.3, 0.5)
DRAWS = (0, 1, 2)


def main():
    cfg = ProtocolConfig()
    cfg_id = "%s:%s" % (cfg.name, hashlib.md5(cfg.to_json().encode()).hexdigest()[:8])
    out = rpath(cfg.out_dir, "drift_study.json")
    acquire_lock(out)
    results = resume_records(out)
    done = {(r["spec"], r["base"], r["sigma"], r["fseed"]) for r in results}
    sel = BlindS3Selector(cfg.selector)
    procs = int(os.environ.get("PROCS", "15"))
    pool = Pool(procs, maxtasksperchild=100)
    ev = PhenoEvaluator(cfg, pool)
    for spec in CODES:
        code = Code(spec)
        for base in BASES:
            for s in DRAWS:
                nominal = NoiseField(base, cfg.p_anchor, s, code)
                frames = baseline_frames(code, nominal)
                frames["Cham"] = sel.select(code, nominal)     # selected on NOMINAL
                for sig in SIGMAS:
                    if (spec, base, sig, s) in done:
                        continue
                    noise = base if sig == 0 else "drift:%g:%d:%s" % (sig, s, base)
                    fld = NoiseField(noise, cfg.p_anchor, s, code)
                    rec = dict(spec=spec, base=base, sigma=sig, fseed=s,
                               cfg_id=cfg_id, masks={})
                    for k, f in frames.items():
                        rec["masks"][k] = ev.worst_axis(code, fld, f, "eta").as_dict()
                    m = rec["masks"]
                    bb = min(m[k]["ler"] for k in BASELINES)
                    ch = m["Cham"]["ler"]
                    ok = ch > 0 and not m["Cham"]["unresolved"]
                    rec["gain_bb"] = bb / ch if ok else None
                    print("[%s %-11s sig=%.1f s%d] bb %.3e Cham %.3e gain %s" % (
                        spec, base, sig, s, bb, ch,
                        ("%.2f" % rec["gain_bb"]) if rec["gain_bb"] else "unres"), flush=True)
                    results.append(rec)
                    atomic_json_dump(results, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    main()
