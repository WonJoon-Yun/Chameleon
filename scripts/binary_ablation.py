"""Binary-vs-6^n frame-space ablation (fig_sixframe refresh, protocol v1.4).
Representative codes x 4 fields x 3 maps at the anchor: baselines + the binary-restricted
selection (top-1 of the {I,H}^n CEM) + the full 6^n selection, all measured identically (anchor p, eta-tier budget).
Usage: PROCS=10 python3 scripts/binary_ablation.py
-> results/protocol_v1/sixframe_typea.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

from chameleon.config import BASELINES
import numpy as np
from multiprocessing import Pool

from chameleon.config import ProtocolConfig
from chameleon.core import Code, NoiseField, Frame, BlindS3Selector, PhenoEvaluator, baseline_frames
from chameleon import search as _search
from chameleon._root import rpath, atomic_json_dump, acquire_lock
from chameleon.records import resume_records

CODES = ("surf2d:5", "color2d:5", "BB72")
DRAWS = (0, 1, 2)


def main():
    cfg = ProtocolConfig.default()
    out = rpath(cfg.out_dir, "sixframe_typea.json")   # the name the exporter reads
    os.makedirs(os.path.dirname(out), exist_ok=True)
    acquire_lock(out)
    pool = Pool(int(os.environ.get("PROCS", "10")), maxtasksperchild=100)
    results = resume_records(out)
    done = {(r["spec"], r["noise"], r["fseed"]) for r in results}
    sel = BlindS3Selector(cfg.selector)
    ev = PhenoEvaluator(cfg, pool)
    jobs = [(spec, f, s) for spec in CODES for f in cfg.fields for s in DRAWS]
    jobs = [j for j in jobs if j not in done]
    print("binary-ablation cells to run:", len(jobs), flush=True)
    for spec, noise, fseed in jobs:
        code = Code(spec)
        fld = NoiseField(noise, cfg.p_anchor, fseed, code)
        U6 = sel.surrogate(code, fld)
        frames = baseline_frames(code, fld)
        # binary-restricted selection: {I,H}^n CEM top-1
        Ub = lambda D: U6(np.atleast_2d(D).astype(int))
        warm = np.array(_search.cem_pool(Ub, code.n, iters=cfg.selector.warm_iters,
                                         M=cfg.selector.warm_pop, seed=cfg.selector.cem_seed)[0][1], int)
        frames["Cham2"] = Frame.from_binary(warm)
        frames["Cham6"] = sel.select(code, fld)
        rec = dict(spec=spec, noise=noise, fseed=fseed, p=cfg.p_anchor, tier="ablation",
                   model=cfg.model, masks={})
        for nm, fr in frames.items():
            rec["masks"][nm] = ev.worst_axis(code, fld, fr, tier="eta").as_dict()
        bb = min(rec["masks"][k]["ler"] for k in BASELINES)
        c6 = rec["masks"]["Cham6"]["ler"]; c2 = rec["masks"]["Cham2"]["ler"]
        print("[%s %-11s s%d] bb %.3e Cham2 %.3e Cham6 %.3e (6/2 = %.2f)" % (
            spec, noise, fseed, bb, c2, c6, (c2 / c6) if c6 > 0 else float("nan")), flush=True)
        results.append(rec); atomic_json_dump(results, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    main()
