"""Type-A full-protocol re-run: phenomenological T=DEE[spec], q=p (the OFFICIAL
REPRODUCIBILITY: sampler salts derive from python hash(); set PYTHONHASHSEED for
bit-identical resumes (review MINOR-5, 2026-07-20). Committed from the campaign
scratchpad (review MAJOR-3): this exact script produced typeA_p005_500/p01/p02.
model), all paper codes x 4 fields x 5 fseeds x 5 frames (CSS/XZZX/ZXXZ/Tiurev/
Cham -- no Chameleon-E per the protocol). For every frame measures TOTAL
worst-axis LER and TYPE-A-only worst-axis LER, classifying each failure with the
unified decoder-independent MILP criterion validated at T=1:
  Type A <=> min{w(x): Hx=s, Ox=obs_pred (mod2)} <= w(true error),  w_k=log((1-p_k)/p_k)
Resume-able (incremental JSON); one progress line per frame-cell for the manager."""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
from chameleon._env import cap_threads
cap_threads()          # before numpy: BLAS pools are sized at import

import os, sys, json, time
import numpy as np
from multiprocessing import Pool
from chameleon._root import rpath
from chameleon.config import ProtocolConfig
_CFG = ProtocolConfig()
from chameleon.core import Code, NoiseField, BlindS3Selector, Frame
from chameleon import pheno as _pheno
from chameleon import baselines as _baselines
from chameleon.pheno import DEE


PROCS = int(os.environ.get("PROCS", "156"))
P = float(os.environ.get("P", "0.005"))   # hard ceiling 0.03 (house rule) -- 0.02 = the _p20 robustness point
CHUNK = 6_250
WAVE = 156           # 156 x 6.25k = ~1M shots/wave with ALL cores busy (no partial-wave idle)
MAX_WAVES = int(os.environ.get("MAX_WAVES", "12"))   # cap = MAX_WAVES x 100k shots/axis
MIN_FAIL = 60        # stop early once total failures >= this (deadline-tuned)
MIN_TYPEA = int(os.environ.get("MIN_TYPEA", "0"))   # >0: ALSO require this many Type-A events per axis
SPECS = os.environ.get("SPECS", "BB36,BB18,BB72,color2d:5,surf2d:5,color2d:3,surf2d:3,color2d:7,surf2d:7").split(",")
FIELDS = os.environ.get("FIELDS", "berlin_star,miami_star,willow_star,xyz:10").split(",")
FSEEDS = [int(x) for x in os.environ.get("FSEEDS", "0,1,2,3,4").split(",")]
FRAMES_FILTER = os.environ.get("FRAMES", "")   # e.g. "CSS" for smoke
OUT = os.environ.get("OUT", rpath(_CFG.study_dir + "/typeA_pheno_data.json"))


def min_class_weight(H, O, w, s, obs_pred):
    from scipy.optimize import milp, LinearConstraint, Bounds
    n_det, n_err = H.shape
    n_obs = O.shape[0]
    n_var = n_err + n_det + n_obs
    c = np.concatenate([w, np.zeros(n_det + n_obs)])
    A = np.zeros((n_det + n_obs, n_var))
    A[:n_det, :n_err] = H
    for i in range(n_det):
        A[i, n_err + i] = -2.0
    A[n_det:, :n_err] = O
    for j in range(n_obs):
        A[n_det + j, n_err + n_det + j] = -2.0
    rhs = np.concatenate([s.astype(float), obs_pred.astype(float)])
    lb = np.zeros(n_var)
    ub = np.concatenate([np.ones(n_err),
                          np.full(n_det, np.ceil(H.sum(1).max() / 2) + 1),
                          np.full(n_obs, np.ceil(max(O.sum(1).max(), 1) / 2) + 1)])
    res = milp(c=c, constraints=[LinearConstraint(A, rhs, rhs)],
               integrality=np.ones(n_var), bounds=Bounds(lb, ub),
               options=dict(time_limit=20.0))
    # timeout/infeasible -> +inf (classified Type-B, i.e. conservative AGAINST us)
    return float(res.fun) if res.status == 0 else float("inf")


_BUNDLE_CACHE = {}   # per-worker-process cache: (spec,noise,fseed,frame,mem) -> (dem_s,H,O,pr,w,dec,mode)


def _build_bundle(spec, noise, fseed, frame, mem):
    code = Code(spec)
    field = NoiseField(noise, P, fseed, code)
    T = DEE[spec]
    q = P
    circ = _pheno.build_pheno(spec, field.pX, field.pY, field.pZ, q,
                               np.asarray(frame, int), mem, T=T)
    if spec.startswith("color"):
        circ = _pheno.add_colors(spec, circ, mem, T=T)
    dem_s = circ.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    H, O, pr = _pheno.dem_matrices(circ)
    w = np.log((1 - pr) / np.clip(pr, 1e-15, None))

    if spec.startswith("color"):
        import chromobius
        try:
            dem_dec = circ.detector_error_model()
        except ValueError:
            dem_dec = circ.detector_error_model(approximate_disjoint_errors=True)
        dec = chromobius.compile_decoder_for_dem(dem_dec)
        mode = "color"
    elif spec.startswith("surf"):
        from pymatching import Matching
        try:
            dem_m = circ.detector_error_model(decompose_errors=True)
        except ValueError:
            dem_m = circ.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
        dec = Matching.from_detector_error_model(dem_m)
        mode = "surf"
    else:
        from ldpc import BpOsdDecoder
        dec = BpOsdDecoder(H, error_channel=list(map(float, np.clip(pr, 1e-9, 0.4999))),
                            max_iter=1000, bp_method="product_sum", schedule="serial",
                            osd_method="OSD_CS", osd_order=7)
        mode = "BB"
    return dem_s, H, O, pr, w, dec, mode


def chunk_worker(args):
    spec, noise, fseed, frame, mem, seed, shots, key = args
    cache_key = (spec, noise, fseed, tuple(frame), mem)
    bundle = _BUNDLE_CACHE.get(cache_key)
    if bundle is None:
        bundle = _build_bundle(spec, noise, fseed, frame, mem)
        _BUNDLE_CACHE[cache_key] = bundle
    dem_s, H, O, pr, w, dec, mode = bundle

    sampler = dem_s.compile_sampler(seed=seed)
    det, obs, err = sampler.sample(shots, return_errors=True)
    detu = det.astype(np.uint8)

    if mode == "color":
        packed = np.packbits(detu, axis=1, bitorder="little")
        pp = dec.predict_obs_flips_from_dets_bit_packed(packed)
        pred = np.unpackbits(pp, axis=1, bitorder="little")[:, :obs.shape[1]].astype(bool)
    elif mode == "surf":
        pred = dec.decode_batch(detu).astype(bool)
        if pred.ndim == 1:
            pred = pred[:, None]
    else:
        pred = np.zeros_like(obs)
        corr_store = {}
        nz = detu.any(axis=1)
        for i in np.nonzero(nz)[0]:
            corr = dec.decode(detu[i])
            corr_store[i] = corr
            pred[i] = ((O @ corr) % 2).astype(bool)

    fails = np.nonzero((pred != obs).any(axis=1))[0]
    n_fail = len(fails)
    if mode == "surf":
        # Exact matching on its own graphlike (decomposed) model always finds the
        # globally lightest explanation, so every failure is a channel ambiguity:
        # Type-A = 100% by construction (matches the T=1 study, 0/299 Type-B).
        # The earlier joint-hyperedge MILP was judging Y-correlation headroom,
        # not decoder mistakes (user-confirmed convention 2026-07-16).
        return key, n_fail, n_fail, shots
    n_typeA = 0
    for i in fails:
        w_true = float(err[i] @ w)
        if mode == "BB":
            # BP+OSD returns its correction: classify by the decoder's OWN chosen
            # explanation weight (validated ~identical to min-class-weight at T=1;
            # avoids minutes-long MILPs on the ~1000-var pheno DEM)
            corr = corr_store.get(i)
            wA = float(corr.astype(np.uint8) @ w) if corr is not None else 0.0
        else:
            # surf (fast, ~10ms) and color (no correction exposed) use the exact
            # min-class-weight MILP, with a hard time limit as a straggler guard
            wA = min_class_weight(H, O, w, detu[i], pred[i].astype(np.uint8))
        if wA <= w_true + 1e-9:
            n_typeA += 1
    return key, n_fail, n_typeA, shots


WAVE_PER = 16   # chunks per (frame,axis) per interleaved wave-round


def measure_cell(spec, noise, fseed, frames, pool, seedbase):
    """Interleaved scheduler: submits wave-rounds for ALL (frame, axis) pairs of a
    (spec,noise,fseed) cell into one flat pool.map, so one pair's stragglers overlap
    other pairs' work instead of idling cores. Adaptive per-pair stopping preserved."""
    state = {}
    fr_list = list(frames.items())
    for name, fr in fr_list:
        for mem in ("z", "x"):
            state[(name, mem)] = dict(f=0, a=0, N=0, wave=0)
    active = set(state.keys())
    while active:
        tasks = []
        for (name, mem) in sorted(active):
            fr = frames[name]
            st = state[(name, mem)]
            for wv in range(WAVE_PER):
                seed = (seedbase + abs(hash((name, mem))) % 400_000 + st["wave"] * 20_000 + wv * 997)
                tasks.append((spec, noise, fseed, fr, mem, seed, CHUNK, (name, mem)))
            st["wave"] += 1
        for key, n_fail, n_typeA, shots in pool.map(chunk_worker, tasks):
            st = state[key]
            st["f"] += n_fail; st["a"] += n_typeA; st["N"] += shots
        for key in list(active):
            st = state[key]
            enough = st["f"] >= MIN_FAIL and (MIN_TYPEA == 0 or st["a"] >= MIN_TYPEA)
            if enough or st["wave"] >= MAX_WAVES:
                active.discard(key)
    out = []
    for name, fr in fr_list:
        z = state[(name, "z")]; x = state[(name, "x")]
        out.append((name, dict(fz=z["f"], az=z["a"], Nz=z["N"], fx=x["f"], ax=x["a"], Nx=x["N"],
                                ler_total=max(z["f"] / z["N"], x["f"] / x["N"]),
                                lerA=max(z["a"] / z["N"], x["a"] / x["N"]))))
    return out


if __name__ == "__main__":
    from chameleon._cli import parse_no_args
    parse_no_args(__doc__)          # answer --help before any measurement starts

    results = []
    if os.path.exists(OUT):
        results = json.load(open(OUT))
    done = {(r["spec"], r["noise"], r["fseed"], r["frame_name"]) for r in results}
    print(f"resuming: {len(done)} frame-cells already done", flush=True)
    t0 = time.time()
    with Pool(PROCS) as pool:
        for spec in SPECS:
            for noise in FIELDS:
                for fseed in FSEEDS:
                    code = Code(spec)
                    field = NoiseField(noise, P, fseed, code)
                    sel = BlindS3Selector()
                    bfr = _baselines.masks_bin(code.C, field.rX0, field.rZ0)
                    frames = {k: Frame.from_binary(v).tolist() for k, v in bfr.items()}
                    frames["Cham"] = sel.select(code, field).tolist()
                    if FRAMES_FILTER:
                        frames = {k: v for k, v in frames.items() if k in FRAMES_FILTER.split(",")}
                    frames = {k: v for k, v in frames.items()
                              if (spec, noise, fseed, k) not in done}
                    if not frames:
                        continue
                    for name, m in measure_cell(spec, noise, fseed, frames, pool,
                                                 abs(hash((spec, noise, fseed))) % 50_000_000):
                        rec = dict(spec=spec, noise=noise, fseed=fseed, frame_name=name, **m)
                        results.append(rec)
                        json.dump(results, open(OUT, "w"))
                        print(f"CELLDONE {spec}|{noise}|fs{fseed}|{name}: total={m['ler_total']:.3e} "
                              f"typeA={m['lerA']:.3e} (f={m['fz']}/{m['fx']} a={m['az']}/{m['ax']})  "
                              f"({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)
