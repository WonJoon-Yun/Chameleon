"""PROTOCOL runner — thin CLI over chameleon.core.Matrix.

Usage:
  GROUP=A PROCS=55 python3 scripts/run_protocol.py
  python3 scripts/run_protocol.py --group B --procs 50 --config myvariant.json

Groups: A = surface+color, B = BB72+BB108, C = BB144. Resumable per
(spec, noise, p, fseed, tier); one output json per group under cfg.out_dir.
Every numeric choice lives in chameleon.config.ProtocolConfig; pass --config to
run any variant without code edits.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

from chameleon.config import BASELINES
import os, sys, argparse
from multiprocessing import Pool

from chameleon.config import ProtocolConfig
from chameleon.core import Matrix
from chameleon._root import rpath, atomic_json_dump, acquire_lock
from chameleon.records import resume_records

# Execution groups: the matrix is sharded so the three can run on separate nodes.
# Membership must COVER the deployed matrix (config.codes) -- a code in no group
# is never measured by `make experiments`, which loops A, B, C. BB18 and BB36
# were deployed but ungrouped and so were silently skipped. Retired codes
# (BB108, BB144, color2d:7) stay listed so an older config still shards, and
# select nothing under the current one.
GROUPS = {"A": ("surf2d:3", "surf2d:5", "surf2d:7", "color2d:3", "color2d:5", "color2d:7"),
          "B": ("BB18", "BB36", "BB72", "BB108"),
          "C": ("BB144",)}


def main():
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").strip().split("\n")[0],
        epilog="\n".join((__doc__ or "").strip().split("\n")[1:]).strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default=os.environ.get("GROUP", "A"), choices=list(GROUPS))
    ap.add_argument("--procs", type=int, default=int(os.environ.get("PROCS", "60")))
    ap.add_argument("--config", default=None,
                    help="ProtocolConfig json (default: the deployed configuration)")
    ap.add_argument("--tier", default=os.environ.get("CHAM_TIERS"),
                    help="comma-separated tiers to run (anchor, dist, eta, psweep); "
                         "default: every tier in the matrix")
    ap.add_argument("--out", default=None, metavar="NAME.json",
                    help="record filename inside the results directory "
                         "(default: protocol_v1_<group>.json). Some shipped records "
                         "hold a tier-restricted slice under their own name, e.g. "
                         "--tier psweep --out xyz_psweep_ext.json")
    ap.add_argument("--strict-resume", action="store_true",
                    help="refuse to run at all when the existing records carry a "
                         "different config, instead of writing to a separate file")
    args = ap.parse_args()

    cfg = ProtocolConfig.from_json(args.config) if args.config else ProtocolConfig.default()
    import hashlib
    cfg_id = "%s:%s" % (cfg.name, hashlib.md5(cfg.to_json().encode()).hexdigest()[:8])
    if args.out is not None:
        if os.path.basename(args.out) != args.out or not args.out.endswith(".json"):
            raise SystemExit("--out takes a bare filename ending in .json, not a path: %r "
                             "(records always go into %s)" % (args.out, cfg.out_dir))
    out = rpath(cfg.out_dir, args.out or "protocol_v1_%s.json" % args.group)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    acquire_lock(out)

    if args.group not in GROUPS:
        raise SystemExit("unknown execution group %r; valid groups are %s "
                         "(set GROUP or pass --group)"
                         % (args.group, ", ".join(sorted(GROUPS))))
    codes = [c for c in cfg.codes if c in GROUPS[args.group]]
    cells = Matrix(cfg).cells(codes)
    if args.tier:
        allow = {t.strip() for t in args.tier.split(",") if t.strip()}
        unknown = allow - set(Matrix.TIERS)
        if unknown:
            raise SystemExit("unknown tier(s) %s; valid tiers are %s"
                             % (", ".join(sorted(unknown)), ", ".join(Matrix.TIERS)))
        cells = [c for c in cells if c.tier in allow]
    # SURFACE d7 dist/eta/psweep are MC-infeasible at p<=0.005 (LER ~4e-6 << event floor within
    # the shot cap; measured 12% resolved) AND the figure renders surf-d7 at anchor tier only ->
    # gate to anchor to save compute and unblock the eta/psweep tiers. NOTE: color-d7 (LER ~2e-5)
    # resolves ~87% and IS used as a violin, so the gate is SURFACE-d7-only, not all ":7".
    # Opt-in (default off preserves prior behavior); does not touch the hashed config (cfg_id
    # unchanged). See results/quality_loop/surf_d7_dist_waste.md
    if os.environ.get("CHAM_D7_ANCHOR_ONLY"):
        cells = [c for c in cells if not (c.spec == "surf2d:7" and c.tier != "anchor")]
    results = resume_records(out)
    # Records are only resumable within one configuration: cells measured under a
    # different config are a different experiment, not a partial run of this one.
    # The shipped records were produced under an earlier revision, so a fresh
    # re-measurement writes alongside them rather than refusing or overwriting.
    prev_ids = {r.get("cfg_id") for r in results if r.get("cfg_id")}
    if prev_ids and prev_ids != {cfg_id}:
        if args.strict_resume:
            raise SystemExit(
                "refusing to resume: %s holds config %s, current is %s.\n"
                "  Drop --strict-resume to re-measure into a separate file instead."
                % (out, ", ".join(sorted(prev_ids)), cfg_id))
        fresh = out[:-len(".json")] + ".%s.json" % cfg_id.replace(":", "_")
        print("existing records in %s carry config %s, this run is %s"
              % (os.path.basename(out), ", ".join(sorted(prev_ids)), cfg_id), flush=True)
        print("re-measuring from scratch into %s (the shipped records are left "
              "untouched; --strict-resume refuses instead)"
              % os.path.basename(fresh), flush=True)
        out = fresh
        results = resume_records(out)
    done = {(r["spec"], r["noise"], r["p"], r["fseed"], r["tier"]) for r in results}
    cells = [c for c in cells if (c.spec, c.noise, c.p, c.fseed, c.tier) not in done]
    print("protocol %s [%s]: %d cells to run -> %s" % (cfg.name, args.group, len(cells), out), flush=True)

    pool = Pool(args.procs, maxtasksperchild=100)
    for c in cells:
        rec = c.run(pool); rec["cfg_id"] = cfg_id
        m = rec["masks"]
        bb = min(m[k]["ler"] for k in BASELINES)
        unres = any(v["unresolved"] for v in m.values())
        print("[%s %-11s p=%.4g s%d %s] CSS %.3e bb %.3e Cham %.3e xBB %s ev=%d%s" % (
            c.spec, c.noise, c.p, c.fseed, c.tier, m["CSS"]["ler"], bb, m["Cham"]["ler"],
            ("%.2f" % rec["gain_bb"]) if rec["gain_bb"] else "n/a",
            m["Cham"]["ev"], " UNRESOLVED" if unres else ""), flush=True)
        results.append(rec); atomic_json_dump(results, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
