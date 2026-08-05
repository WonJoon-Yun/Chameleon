"""Device-placement study under the protocol noise model (pheno, q=p, T=d), on the
raw calibration maps (no resampling). OO pipeline: Code / Frame / PhenoEvaluator.

Willow is mean-rescaled to the p=0.005 anchor (labeled); q = patch mean total rate.
Usage: PROCS=12 python3 scripts/patch_pheno.py
-> results/protocol_v1/patch_pheno.json
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

from chameleon.config import ProtocolConfig
_CFG = ProtocolConfig()
from chameleon.config import BASELINES
import os, sys, json, math, csv
import numpy as np
from multiprocessing import Pool

import chameleon as ch
from chameleon.config import ProtocolConfig
from chameleon.core import Code, BlindS3Selector, baseline_frames
from chameleon._root import rpath, atomic_json_dump
from chameleon.records import resume_records


def ibm_sites(dev):
    T1 = {}; T2 = {}; rd = {}
    for r in csv.DictReader(open(rpath(ch.fields.CSV["ibm_" + dev]))):
        try:
            q = int(r["Qubit"]); T1[q] = float(r["T1 (us)"]); T2[q] = float(r["T2 (us)"])
            rd[q] = float(r["Readout length (ns)"]) / 1000.0
        except ValueError:
            pass  # blank/malformed calibration cell -> skip this qubit
    t = float(np.median(list(rd.values())))
    Pm = {}
    for q in T1:
        pxy = (1 - math.exp(-t / T1[q])) / 4.0
        pz = max((1 - math.exp(-t / T2[q])) / 2.0 - pxy, 0.0)
        Pm[q] = [pxy, pxy, pz]
    cz = json.load(open(rpath(_CFG.pool_dir, "fakechar_ibm_%s.json" % dev)))["couplers"]
    for k, chn in cz.items():
        _, a, b = k.split("_"); a, b = int(a), int(b)
        for pstr, v in chn.items():
            for qi, sym in [(a, pstr[0]), (b, pstr[1])]:
                if qi in Pm and sym != "I":
                    Pm[qi]["XYZ".index(sym)] += v
    return Pm


def willow_sites():
    cal = json.load(open(rpath("calibration/willow_calibration.json")))
    Pm = {}
    for k, v in cal["per_qubit"].items():
        r, c = (int(x) for x in k[2:-1].split(", "))
        pxy = (1 - math.exp(-25.0 / v["T1_ns"])) / 4.0
        pz = max((1 - math.exp(-25.0 / v["T2_ns"])) / 2.0 - pxy, 0.0)
        Pm[(r, c)] = [pxy, pxy, pz]
    return Pm


def footprints(sites, size):
    """size x size data footprints on the site grid (tuple-coordinate or r*10+c indexing)."""
    keys = list(sites)
    pats = []
    if isinstance(keys[0], tuple):
        rs = [r for r, _ in keys]; cs = [c for _, c in keys]
        for r in range(min(rs), max(rs) - size + 2):
            for c in range(min(cs), max(cs) - size + 2):
                w = [(r + i, c + j) for i in range(size) for j in range(size)]
                if all(q in sites for q in w):
                    pats.append(tuple(w))
        return pats
    for r in range(12):
        for c in range(10):
            w = [(r + i) * 10 + (c + j) if (0 <= r + i < 12 and 0 <= c + j < 10) else None
                 for i in range(size) for j in range(size)]
            if all(q is not None and q in sites for q in w):
                pats.append(tuple(w))
    return pats


class RawPatchField:
    """A NoiseField-compatible object built from raw per-site triples (no resampling)."""

    def __init__(self, p3, noise, seed=0):
        p3 = np.asarray(p3, float)
        self.pX, self.pY, self.pZ = p3[:, 0], p3[:, 1], p3[:, 2]
        self.p = float(p3.sum(1).mean())     # the q = p convention on a raw map
        self.noise, self.seed = noise, seed

    @property
    def rX0(self):
        return self.pX + self.pY

    @property
    def rZ0(self):
        return self.pZ + self.pY


def worst_axis_raw(pool, cfg, code, fld, frame, minev=60):
    """Protocol sampling schedule on a raw field (channels passed explicitly)."""
    s = cfg.sampling; cap = cfg.tiers["anchor"].cap
    fX = fZ = N = 0; wave = 0; cz = []; cx = []
    while N < cap:
        args = [(mem, s.chunk_seed(wave, k, mem), s.per)
                for mem in ("z", "x") for k in range(s.nch)]
        res = pool.map(_raw_chunk, [(code.spec, fld.pX.tolist(), fld.pY.tolist(), fld.pZ.tolist(),
                                     fld.p, frame.tolist()) + a for a in args])
        cz += [f for f, _ in res[:s.nch]]; cx += [f for f, _ in res[s.nch:]]
        fZ += sum(f for f, _ in res[:s.nch]); fX += sum(f for f, _ in res[s.nch:])
        N += s.per * s.nch; wave += 1
        if max(fX, fZ) >= minev:
            break
    m = max(fX, fZ)
    return m / N, int(m), N, cz, cx


def _raw_chunk(args):
    spec, pXl, pYl, pZl, q, perm, mem, sseed, shots = args
    from pymatching import Matching
    import chameleon.pheno as _pheno
    circ = _pheno.build_pheno(spec, np.array(pXl), np.array(pYl), np.array(pZl), q,
                              np.asarray(perm, int), mem)
    dem = circ.detector_error_model(decompose_errors=True)
    mch = Matching.from_detector_error_model(dem)
    det, ob = circ.compile_detector_sampler(seed=sseed).sample(shots, separate_observables=True)
    pred = mch.decode_batch(det)
    return int((pred != ob).any(axis=1).sum()), shots


def main():
    cfg = ProtocolConfig.default()
    out = rpath(cfg.out_dir, "patch_pheno.json")
    pool = Pool(int(os.environ.get("PROCS", "12")), maxtasksperchild=100)
    results = resume_records(out)
    done = {(r["spec"], r["dev"], r["idx"]) for r in results}
    sel = BlindS3Selector(cfg.selector)
    jobs = []
    for dev in ("berlin", "miami", "willow"):
        Pm = willow_sites() if dev == "willow" else ibm_sites(dev)
        for spec, size in (("surf2d:3", 3), ("surf2d:5", 5)):
            for idx, pat in enumerate(footprints(set(Pm), size)):
                p3 = np.array([Pm[q] for q in pat])
                if dev == "willow":
                    p3 = p3 * (0.005 / p3.sum(1).mean())
                jobs.append((spec, dev, idx, p3))
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print("patch-pheno cells to run:", len(jobs), flush=True)
    for spec, dev, idx, p3 in jobs:
        code = Code(spec)
        fld = RawPatchField(p3, noise="%s_raw" % dev)
        frames = baseline_frames(code, fld)
        frames["Cham"] = sel.select(code, fld)
        rec = dict(spec=spec, dev=dev, idx=idx, q=fld.p, masks={})
        for nm, fr in frames.items():
            ler, ev, N, cz, cx = worst_axis_raw(pool, cfg, code, fld, fr)
            rec["masks"][nm] = dict(ler=ler, ev=ev, N=N, unresolved=bool(ev < 60), cz=cz, cx=cx)
        bb = min(rec["masks"][k]["ler"] for k in BASELINES)
        c = rec["masks"]["Cham"]["ler"]
        rec["gain_bb"] = bb / c if c > 0 else None
        print("[%s %s p%d] CSS %.3e bb %.3e Cham %.3e xBB %s ev=%d" % (
            spec, dev, idx, rec["masks"]["CSS"]["ler"], bb, c,
            ("%.2f" % rec["gain_bb"]) if rec["gain_bb"] else "n/a",
            rec["masks"]["Cham"]["ev"]), flush=True)
        results.append(rec); atomic_json_dump(results, out)
    print("DONE", flush=True)


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    main()
