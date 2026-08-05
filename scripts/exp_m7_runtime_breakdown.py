"""M7 -- MEASURED end-to-end compile cost, stage by stage, on ONE machine.

The paper's "days -> minutes" headline currently rests on a PROJECTION that assumes an
idealised 1 us/shot decoder.  This script replaces the projection with measurements of
the real thing on the same host:

  once per code   : ambiguity enumeration, LUT construction
  per map         : binary CEM, S3 CEM  (the deployed schedule)
  once/deployment : decoder build (stim circuit -> DEM -> compiled decoder)
  the alternative : ACTUAL decoder throughput (s/shot) of the deployed decoder, hence
                    the real wall time of scoring one candidate frame to a fixed event
                    floor, and of a 24-candidate decode-validated search (the CDSC budget)

and reports the amortisation T_avg(M) = T_enum/M + T_search over M calibration maps.

Peak RSS is sampled per stage with resource.getrusage(RUSAGE_SELF).ru_maxrss (high-water
mark, so it is reported as a monotone envelope, not a per-stage delta).

Outputs results/typeA_investigation/m7_runtime_breakdown.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

import os, sys, json, time, argparse, resource
import numpy as np

from chameleon._root import rpath
from chameleon.config import ProtocolConfig
_CFG = ProtocolConfig()
from chameleon.codes import get_code
from chameleon.mechs import mechs
from chameleon.fields import field3x
from chameleon.surrogate import build_U6
from chameleon.search import cem_pool, cem6
from chameleon import core as _core
from chameleon.config import ProtocolConfig

OUT = rpath(_CFG.study_dir + "/m7_runtime_breakdown.json")


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Stage:
    def __init__(self, d, name):
        self.d, self.name = d, name

    def __enter__(self):
        self.t0, self.c0 = time.perf_counter(), cpu_s(); return self

    def __exit__(self, *a):
        self.d[self.name] = dict(wall_s=time.perf_counter() - self.t0,
                                 cpu_s=cpu_s() - self.c0, peak_rss_MB=rss_mb())


def decoder_throughput(spec, noise, fseed, p, shots, cfg):
    """Measured s/shot of the DEPLOYED decoder on this host, plus the decoder build time."""
    dcfg = dict(bp_iters=cfg.decoder.bp_iters, bp_method=cfg.decoder.bp_method,
                schedule=getattr(cfg.decoder, "schedule", "serial"),
                ms_scaling_factor=getattr(cfg.decoder, "ms_scaling_factor", 1.0),
                osd_method=cfg.decoder.osd_method, osd_order=cfg.decoder.osd_order,
                q_ratio=getattr(cfg, "q_ratio", 1.0))
    C, _, _ = get_code(spec)
    n = C["n"]
    perm = np.zeros(n, int)
    _core._CHUNK_CACHE.clear()
    t0 = time.perf_counter()
    _core._chunk_setup(spec, noise, fseed, p, perm, "z", dcfg)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    f, N = _core._decode_chunk((spec, noise, fseed, p, perm, "z", 12345, shots, dcfg))
    t_dec = time.perf_counter() - t0
    return dict(decoder_build_s=t_build, shots=int(N), decode_wall_s=t_dec,
                s_per_shot=t_dec / max(N, 1), failures=int(f))


def run(spec, noise, fseed, p, shots, floor_events, cdsc_candidates, search_cands):
    cfg = ProtocolConfig()
    d = dict(spec=spec, noise=noise, fseed=fseed, p=p, stages={})
    C, _, _ = get_code(spec)
    n = C["n"]; d["n"] = n

    with Stage(d["stages"], "enumeration"):
        LX, LZ = mechs(spec, C)
    d["n_mech_X"], d["n_mech_Z"] = len(LX), len(LZ)

    pX, pY, pZ = field3x(noise, n, fseed, p)
    s = (pX + pY + pZ).mean(); sc = p / max(s, 1e-15)
    pX, pY, pZ = pX * sc, pY * sc, pZ * sc

    with Stage(d["stages"], "lut_build"):
        U = build_U6(LX, LZ, n, pX, pY, pZ)
        U(np.zeros((8, n), int))                    # force the LUT/broadcast path
    d["lut_MB"] = (len(LX) + len(LZ)) * n * 8 / 2 ** 20

    with Stage(d["stages"], "binary_cem"):
        Ub = lambda S: U(np.where(np.asarray(S, bool), 1, 0))
        warm = np.where(np.asarray(cem_pool(Ub, n, iters=cfg.selector.warm_iters,
                                            M=cfg.selector.warm_pop,
                                            seed=cfg.selector.cem_seed)[0][1], bool), 1, 0)
    with Stage(d["stages"], "s3_cem"):
        F = np.asarray(cem6(U, n, iters=cfg.selector.cem_iters, M=cfg.selector.cem_pop,
                            seed=cfg.selector.cem_seed, warm=warm), int)
    d["frame"] = [int(x) for x in F]

    d["decoder"] = decoder_throughput(spec, noise, fseed, p, shots, cfg)

    # --- derived: what the decode-in-the-loop alternative actually costs here
    sps = d["decoder"]["s_per_shot"]
    ler = max(d["decoder"]["failures"], 1) / max(d["decoder"]["shots"], 1)
    shots_to_floor = floor_events / max(ler, 1e-12)
    d["derived"] = dict(
        measured_LER_at_identity=ler,
        shots_for_floor=shots_to_floor, floor_events=floor_events,
        s_per_candidate_to_floor=shots_to_floor * sps * 2.0,          # two memory bases
        cdsc_candidates=cdsc_candidates,
        cdsc_wall_s=cdsc_candidates * shots_to_floor * sps * 2.0,
        search_candidates=search_cands,
        full_decode_search_wall_s=search_cands * shots_to_floor * sps * 2.0,
        chameleon_per_map_s=d["stages"]["binary_cem"]["wall_s"] + d["stages"]["s3_cem"]["wall_s"],
        once_per_code_s=d["stages"]["enumeration"]["wall_s"] + d["stages"]["lut_build"]["wall_s"])
    per_map = d["derived"]["chameleon_per_map_s"]
    once = d["derived"]["once_per_code_s"]
    d["amortised_s_per_map"] = {str(M): once / M + per_map for M in (1, 10, 100, 1000)}
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="surf2d:3,surf2d:5,surf2d:7,color2d:5,color2d:7,"
                                       "BB18,BB36,BB72")
    ap.add_argument("--noise", default="willow_star")
    ap.add_argument("--fseed", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--floor-events", type=int, default=500)
    ap.add_argument("--cdsc", type=int, default=24)
    ap.add_argument("--search-cands", type=int, default=78_000)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    rows = []
    for spec in a.specs.split(","):
        try:
            r = run(spec, a.noise, a.fseed, a.p, a.shots, a.floor_events,
                    a.cdsc, a.search_cands)
        except Exception as e:
            print("FAIL %s: %r" % (spec, e), flush=True); continue
        rows.append(r)
        st = r["stages"]; dv = r["derived"]
        print("%-10s enum %7.2fs  lut %6.3fs  binCEM %6.2fs  S3CEM %6.2fs | decoder build %6.2fs"
              "  %.3e s/shot | per-map %.2fs vs CDSC-%d %.1f h vs full-decode %.0f h"
              % (spec, st["enumeration"]["wall_s"], st["lut_build"]["wall_s"],
                 st["binary_cem"]["wall_s"], st["s3_cem"]["wall_s"],
                 r["decoder"]["decoder_build_s"], r["decoder"]["s_per_shot"],
                 dv["chameleon_per_map_s"], a.cdsc, dv["cdsc_wall_s"] / 3600,
                 dv["full_decode_search_wall_s"] / 3600), flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)
    json.dump(rows, open(a.out, "w"), indent=1)
    print("\nwrote %s (%d codes)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
