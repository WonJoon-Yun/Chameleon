from .config import ProtocolConfig
_CFG = ProtocolConfig()
"""Failure-mechanism enumeration under the 100% full-set rule (v1.6+; 95%-mass abolished) (verbatim from cham_lib.logops /
enriched_mechs and pheno_full_suite.mechs; BB orbit enrichment via experiments.bb_mech_orbit)."""
import sys, json, itertools
import numpy as np
from ._root import ROOT, rpath

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_ENRICHED_CACHE = {}
_BB_MECH_CACHE = {}

# Verified full-set (X, Z) counts for every precomputed enriched spec (experiment-
# audit). A stale or short-regenerated enriched_mechs.json would
# silently feed the surrogate a truncated mechanism family -- the exact failure mode
# of the BB36 bug -- so enriched_mechs() fails loud on any mismatch at read time.
_ENRICHED_EXPECT = {
    "color2d:5": (111, 111),
    "color2d:7": (4078, 4078),
    "surf2d:7": (2497, 2497),
    "BB36": (996, 996),
}


def logops(H, L, n, wcut=None, extra=2, cap=4000, scancap=4_000_000, chunk=200000, strict=False):
    """Enumerate syndrome-free nontrivial mechanisms weight-ordered.
    wcut: absolute max weight (overrides d_min+extra)."""
    Ht = H.T.astype(np.uint8) % 2; Lt = L.T.astype(np.uint8) % 2
    ops = []; first = None; wc = n; sc = 0
    for w in range(1, n + 1):
        got = False; gen = itertools.combinations(range(n), w); done = False; wtrunc = False
        while not done:
            buf = list(itertools.islice(gen, chunk))
            if not buf:
                break
            sc += len(buf)
            B = np.zeros((len(buf), n), np.uint8); idx = np.array(buf)
            B[np.arange(len(buf))[:, None], idx] = 1
            keep = (~((B @ Ht) % 2).any(1)) & (((B @ Lt) % 2).any(1))
            for r in np.nonzero(keep)[0]:
                ops.append((w, tuple(buf[r]))); got = True
                if len(ops) >= cap:
                    msg = ("logops: cap=%d hit -- TRUNCATED mechanism set "
                           "(full-set rule violated for this call)" % cap)
                    if strict:
                        raise RuntimeError(msg)   # deployed path: never truncate silently
                    import warnings
                    warnings.warn(msg)
                    return ops
            if sc >= scancap and first is not None:
                wtrunc = True; done = True
        if got and first is None:
            first = w; wc = (wcut if wcut else min(n, w + extra))
        # scancap check FIRST (before the wc break): a weight whose scan was cut
        # short by scancap, with combinations still unscanned, is a real truncation.
        # (Bug fixed 2026-07-14: the old wc break at w==wc fired before this check,
        # so strict never raised and BB36 silently dropped 96% of its weight-8 set.)
        if wtrunc and next(gen, None) is not None:
            msg = ("logops: scancap=%d hit at w=%d, weight-%d scan incomplete "
                   "-- mechanism set TRUNCATED (raise scancap for full-set rule)"
                   % (scancap, w, w))
            if strict:
                raise RuntimeError(msg)   # deployed path: fail loud, never truncate silently
            import warnings
            warnings.warn(msg)
            break
        if first is not None and w >= wc:
            break
    return ops  # list of (weight, support-tuple)


def enriched_mechs(spec):
    """100% full-set rule (v1.6+; 95%-mass abolished): exhaustive low-weight sets (w<=min+2; the color d7 set runs to min+4, a
    <1%-of-U tail) for the codes where enumeration is expensive (color d5/d7, surf d7). Returns (LX,LZ) or None. Data: results/mechs/enriched_mechs.json."""
    if spec not in _ENRICHED_CACHE:
        p = rpath(_CFG.mech_dir, "enriched_mechs.json")
        import os
        if os.path.exists(p):
            with open(p) as _f:
                EM = json.load(_f)
        else:
            EM = {}
        _ENRICHED_CACHE[spec] = (EM[spec]["X"], EM[spec]["Z"]) if spec in EM else None
    val = _ENRICHED_CACHE[spec]
    if val is not None and spec in _ENRICHED_EXPECT:
        got, exp = (len(val[0]), len(val[1])), _ENRICHED_EXPECT[spec]
        if got != exp:
            raise RuntimeError(
                "enriched_mechs(%r): count %s != verified full-set %s -- "
                "enriched_mechs.json is stale or truncated (regenerate the set)"
                % (spec, got, exp))
    return val


def bb_mechs(spec):
    """Full orbit-enriched BB mechanism set at SATURATION depth (v1.9.2): the
    enumeration runs to extra = max-histogram-weight - d + 2, the depth at which
    a further +2 adds <0.5% new representatives on every BB code (measured:
    BB72 45,483/45,567 at extra=8+, BB108 9,603/9,423 at extra=6+, BB144
    247,473/248,157 at extra=10, bit-identical to the recorded full enumeration).
    The earlier w95-derived depth under-enumerated BB144 by 5x and BB72 by 12%."""
    if spec not in _BB_MECH_CACHE:
        from chameleon.vendor.bb_mech_orbit import enrich
        wh = json.load(open(rpath(_CFG.mech_dir, "wt_histogram.json")))[spec]
        wmax = max(wh["weights"])
        d = {"BB72": 6, "BB90": 10, "BB108": 10, "BB144": 12}.get(spec, 6)
        LXe, LZe = enrich(spec, extra=max(2, wmax - d) + 2, perms=800, orbit=True)
        # Truncation floor: the deployed orbit set must not fall below the recorded
        # full-set size (sum over the weight histogram). BB144's deployed set exceeds
        # this floor (247k vs 221k) from orbit expansion, so >= not == ; a set that
        # drops BELOW the floor is a truncated enumeration and fails loud (same silent-
        # truncation class as the BB36 bug).
        floor = sum(wh["Nw"].values())
        if len(LXe) < floor or len(LZe) < floor:
            raise RuntimeError(
                "bb_mechs(%r): orbit set (%d, %d) fell below recorded floor %d -- "
                "truncated enumeration (raise bb_mech_orbit depth/perms)"
                % (spec, len(LXe), len(LZe), floor))
        _BB_MECH_CACHE[spec] = ([list(s) for s in LXe], [list(s) for s in LZe])
    return _BB_MECH_CACHE[spec]


def mechs(spec, C):
    """The deployed mechanism set (v1.6, 100% rule): the FULL enumerated family —
    every mechanism of weight at most w_min+2 for geometric codes (color d7 runs
    to w_min+4, a <1%-of-U tail; precomputed exhaustive sets where enumeration is
    expensive), and the full orbit-enriched set for the BB codes. No mass
    truncation is applied."""
    n = C["n"]
    if spec.startswith("BB"):
        # Route by capability, NOT a hardcoded allowlist: only the low-distance
        # BB18/BB30 fit the exhaustive small-BB scan. Any larger/higher-distance BB
        # code (BB72/90/108/144/...) MUST use the orbit-enriched enumerator, whose
        # cost scales with the retained mechanism count rather than the intractable
        # exhaustive scan its min-weight layer would need (e.g. BB90 d=10 needs
        # C(90,10)~6e12 supports, far past any scancap -> silent truncation). An
        # allowlist that omitted a large code would reintroduce the 2026-07-13
        # scancap-truncation bug, so the small path is gated to {BB18,BB30,BB36}.
        if spec in ("BB18", "BB30", "BB36"):
            em = enriched_mechs(spec)   # BB36 precomputed set (996), skips the ~278s re-enumeration
            if em is not None:
                return [list(c) for c in em[0]], [list(c) for c in em[1]]
            # BB30's w<=8 layer needs ~8.5M support scans; BB36 [[36,4,6]] w<=8 needs
            # C(36,8)=30.3M (~37M cumulative), so it MUST use a 50M cap -- the old 12M
            # silently truncated it to 136 mechs (true set 996, w8 900), fixed 2026-07-14
            # together with the logops scancap/strict ordering bug that hid the truncation.
            return ([list(c) for w, c in logops(C["Hz"], C["LzZ"], n, extra=2,
                                                scancap=50_000_000, cap=20000, strict=True)],
                    [list(c) for w, c in logops(C["Hx"], C["LxX"], n, extra=2,
                                                scancap=50_000_000, cap=20000, strict=True)])
        return bb_mechs(spec)
    em = enriched_mechs(spec)
    if em is not None:
        return [list(c) for c in em[0]], [list(c) for c in em[1]]
    # Fallback for geometric codes not in the enriched JSON (small d only: d>=7
    # needs ~1e9 candidate scans and MUST be precomputed). strict=True so a large
    # code that would truncate at the default scancap fails loudly, never silently.
    return ([list(c) for w, c in logops(C["Hz"], C["LzZ"], n, extra=2, strict=True)],
            [list(c) for w, c in logops(C["Hx"], C["LxX"], n, extra=2, strict=True)])
