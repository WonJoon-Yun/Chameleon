"""Decode-validated selection vs the decode-free blind pass (paper subsection data).

For representative codes x 4 fields x 3 maps at the protocol anchor:
  Cham    = decode-free blind top-1 (the deployed pass)
  ChamMC  = the CEM top-24 pool, each candidate decode-validated at the dist-tier
            budget on a disjoint selection-seed schedule, winner deployed
Both are then measured identically (eta-tier budget, standard seeds), so the pair
quantifies exactly what a 24-candidate decoding budget adds over the blind pass.
Usage: PROCS=10 python3 scripts/decode_validation.py
-> results/protocol_v1/decode_validation.json
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
from chameleon.core import Code, NoiseField, BlindS3Selector, PhenoEvaluator, baseline_frames
from chameleon._root import rpath, atomic_json_dump, acquire_lock
from chameleon.records import resume_records

CODES = ("surf2d:5", "color2d:5", "BB72")
DRAWS = (0, 1, 2)
K = 24


def main():
    cfg = ProtocolConfig.default()
    # disjoint seed schedule for the validation stage (never reused for measurement).
    # chunk_seed = seed0 + 1000*wave + 17*chunk + mem; the largest tier spans < ~30k
    # seed units, so validation (555000) and measurement (default 1100) stay disjoint
    # by > 500k -- guard it so a future seed0 edit cannot silently break held-out.
    vcfg = dataclasses.replace(cfg, sampling=dataclasses.replace(cfg.sampling, seed0=555000))
    assert abs(vcfg.sampling.seed0 - cfg.sampling.seed0) > 100_000, \
        "validation/measurement seed0 too close: held-out independence not guaranteed"
    out = rpath(cfg.out_dir, "decode_validation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    acquire_lock(out)
    pool = Pool(int(os.environ.get("PROCS", "10")), maxtasksperchild=100)
    results = resume_records(out)
    done = {(r["spec"], r["noise"], r["fseed"]) for r in results}
    sel = BlindS3Selector(cfg.selector)
    meas = PhenoEvaluator(cfg, pool)
    val = PhenoEvaluator(vcfg, pool)
    jobs = [(s, f, m) for s in CODES for f in cfg.fields for m in DRAWS]
    jobs = [j for j in jobs if j not in done]
    print("decode-validation cells to run:", len(jobs), flush=True)
    for spec, noise, fseed in jobs:
        code = Code(spec)
        fld = NoiseField(noise, cfg.p_anchor, fseed, code)
        pool24 = sel.pool(code, fld, topk=K)
        # validation stage: dist-tier budget on the selection seeds
        vals = [val.worst_axis(code, fld, fr, tier="dist").ler for fr in pool24]
        best = pool24[int(min(range(len(vals)), key=lambda i: vals[i]))]
        frames = baseline_frames(code, fld)
        frames["Cham"] = pool24[0]          # blind top-1
        frames["ChamMC"] = best             # decode-validated winner
        rec = dict(spec=spec, noise=noise, fseed=fseed, p=cfg.p_anchor,
                   tier="decodeval", model=cfg.model, masks={})
        for nm, fr in frames.items():
            rec["masks"][nm] = meas.worst_axis(code, fld, fr, tier="eta").as_dict()
        bb = min(rec["masks"][k]["ler"] for k in BASELINES)
        c1 = rec["masks"]["Cham"]["ler"]; c2 = rec["masks"]["ChamMC"]["ler"]
        print("[%s %-11s s%d] bb %.3e blind %.3e MC24 %.3e (MC/blind %.2f)" % (
            spec, noise, fseed, bb, c1, c2, (c1 / c2) if c2 > 0 else float("nan")), flush=True)
        results.append(rec); atomic_json_dump(results, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    main()
