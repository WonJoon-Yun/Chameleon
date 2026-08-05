"""Loss post-mortem + compile-time abstention criterion (reviewer-hardening).
For every RESOLVED distribution-tier cell (surf/color d5, measured fields), recompute
the surrogate's predicted margin for the deployed frame,
    margin(map) = min_b U(baseline_b) / U(Cham),
pair it with the measured gain_xfit, and evaluate the abstention rule
"deploy only if margin >= tau": losses avoided vs wins forfeited per tau.
Selection is deterministic, so this is analysis-time only -- no new decoding.
  PROCS=8 python3 scripts/abstention_analysis.py
-> results/protocol_v1/abstention.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

from chameleon.config import ProtocolConfig
_CFG = ProtocolConfig()
import numpy as np
from multiprocessing import Pool

from chameleon.config import ProtocolConfig
from chameleon.core import Code, NoiseField, BlindS3Selector, baseline_frames
import chameleon.surrogate as sur
from chameleon.records import load_protocol, resume_records
from chameleon._root import rpath, atomic_json_dump, acquire_lock

SPECS = ("surf2d:5", "color2d:5")
FIELDS = ("berlin_star", "miami_star", "willow_star", "xyz:10")


def job(args):
    spec, noise, fseed, gain = args
    cfg = ProtocolConfig.default()
    code = Code(spec)
    fld = NoiseField(noise, cfg.p_anchor, fseed, code)
    LX, LZ = code.mechs
    U = sur.build_U6(LX, LZ, code.n, fld.pX, fld.pY, fld.pZ)
    sel = BlindS3Selector(cfg.selector)
    u_ch = float(U(np.asarray(sel.select(code, fld).tolist())[None, :])[0])
    u_bb = min(float(U(np.asarray(fr.tolist())[None, :])[0])
               for nm, fr in baseline_frames(code, fld).items())
    return dict(spec=spec, noise=noise, fseed=fseed, gain_xfit=gain,
                margin=u_bb / u_ch if u_ch > 0 else None)


def main():
    out = rpath(_CFG.out_dir + "/abstention.json")
    acquire_lock(out)
    results = resume_records(out)
    done = {(r["spec"], r["noise"], r["fseed"]) for r in results}
    cells = [(r["spec"], r["noise"], r["fseed"], r["gain_xfit"])
             for r in load_protocol()
             if r["tier"] == "dist" and r["spec"] in SPECS and r["noise"] in FIELDS
             and r.get("gain_xfit") and (r["spec"], r["noise"], r["fseed"]) not in done]
    print("abstention cells:", len(cells), flush=True)
    with Pool(int(os.environ.get("PROCS", "8")), maxtasksperchild=40) as pool:
        for i, rec in enumerate(pool.imap_unordered(job, cells, chunksize=4)):
            results.append(rec)
            if i % 50 == 0:
                print("%d/%d" % (i, len(cells)), flush=True)
                atomic_json_dump(results, out)
    atomic_json_dump(results, out)
    # summary: abstention operating points
    for spec in SPECS:
        rs = [r for r in results if r["spec"] == spec and r["margin"]]
        if not rs:
            continue
        g = np.array([r["gain_xfit"] for r in rs]); m = np.array([r["margin"] for r in rs])
        print("%s n=%d  corr(log margin, log gain)=%.2f" % (
            spec, len(rs), np.corrcoef(np.log(m), np.log(g))[0, 1]), flush=True)
        for tau in (1.0, 1.05, 1.1, 1.2):
            keep = m >= tau
            print("  tau=%.2f deploy %d%%: losses %d->%d, median gain %.2f->%.2f" % (
                tau, 100 * keep.mean(), (g < 1).sum(), (g[keep] < 1).sum(),
                np.median(g), np.median(g[keep]) if keep.any() else float("nan")), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    main()
