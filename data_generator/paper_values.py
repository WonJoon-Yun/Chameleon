"""Emit output/data/paper_values.csv: every reported number as a name,value row
macro computed from the current protocol's data (config.py `name` field), so the text
never hardcodes a value.

Naming: val <Spec> <Field> <Stat>, spelled without digits, e.g.
  \\valSurfFiveXyzGain     anchor-tier mean gain over the best fixed baseline
  \\valSurfFiveXyzGainLo / ...Hi   min / max over the five anchor maps
  \\valSurfFiveXyzMedian   100-map distribution median (anchor+dist)
  \\valSurfFiveXyzLossPct  percent of maps with gain below one
Cells without data yet render as [TBD].
Usage: python3 scripts/make_values.py
"""
import os, sys, json
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))
import argparse, csv, re, contextlib
import numpy as np
from chameleon._root import rpath
from chameleon.config import ProtocolConfig
_CFG = ProtocolConfig()
from chameleon.config import BASELINES
from chameleon.records import OUT_DIR      # honours CHAM_PAPER_DIR, as export_results does

os.makedirs(rpath(OUT_DIR, "data"), exist_ok=True)

P_ANCHOR = 0.005   # canonical operating point; anchor selections must never blend an off-rate record
                   # (e.g. a p=0.01 robustness or p=0.03 near-threshold anchor) into the p=0.005 macros.

SPEC = {"surf2d:3": "SurfThree", "surf2d:5": "SurfFive", "surf2d:7": "SurfSeven",
        "color2d:3": "ColorThree", "color2d:5": "ColorFive", "color2d:7": "ColorSeven",
        "BB72": "BBSeventyTwo", "BB18": "BBEighteen", "BB36": "BBThirtySix",
        "BB108": "BBOneOhEight", "BB144": "BBOneFourFour"}
FIELD = {"berlin_star": "Berlin", "miami_star": "Miami",
         "willow_star": "Willow", "xyz:10": "Xyz"}


def load():
    from chameleon.records import load_protocol
    return load_protocol()


def resolved(recs):
    return [r for r in recs if r.get("gain_bb") is not None]


def fmt(x, nd=2):
    return ("%%.%df" % nd) % x


def fmt_pct(x):
    """Gain ratio -> signed percent-change string, matching Table V's own convention
    (one notation paper-wide): 0.81x -> -19, 1.05x -> +5,
    4.42x -> +342. Callsites in the prose drop the trailing \\times for \\%."""
    return "%+d" % round((x - 1) * 100)


from chameleon.records import _recompute_gains, headline_gain, _XFIT_CHUNK
from chameleon._root import rpath as _rp
import json as _json

def _hg(recs):
    return [headline_gain(r) for r in recs if headline_gain(r)]


class ValueTable:
    """Ordered table of the paper's reported values.

    Sections emit through ``append`` in the historical ``\\newcommand`` form; the
    table parses each entry once, at emission, and stores the (name, value) pair.
    Nothing is re-parsed at write time, and a malformed or empty value is caught
    where it is produced rather than at the end of the run.
    """

    _ENTRY = re.compile(r"\\newcommand\{\\(\w+)\}\{(.*)\}\s*$")
    _BAD = ("", "none", "nan", "inf", "-inf")

    def __init__(self):
        self._rows = []
        self._index = {}
        self.suspicious = []

    def append(self, line):
        """Accept one emitted entry. Comment lines are ignored."""
        if not line.startswith("\\newcommand"):
            return
        m = self._ENTRY.match(line)
        if m is None:
            raise ValueError("malformed value entry: %r" % line)
        name, value = m.group(1), m.group(2)
        if name in self._index and self._index[name] != value:
            raise ValueError("value %r emitted twice with different values: %r vs %r"
                             % (name, self._index[name], value))
        if value.strip().lower().strip("$ ") in self._BAD:
            self.suspicious.append((name, value))
        self._index[name] = value
        self._rows.append((name, value))

    def extend(self, lines):
        for line in lines:
            self.append(line)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    def get(self, name, default=None):
        """Value already emitted under ``name``, or ``default``."""
        return self._index.get(name, default)

    def names(self):
        return set(self._index)

    def matching(self, pattern):
        """(match, value) for every emitted name matching ``pattern``."""
        rx = re.compile(pattern)
        return [(m, self._index[nm]) for nm in self._index
                for m in (rx.fullmatch(nm),) if m]

    def write_csv(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "value"])
            w.writerows(self._rows)


class SectionSkipped(Exception):
    """A section could not be computed from the shipped records."""


@contextlib.contextmanager
def section(name, skipped):
    """Guard one group of values.

    A group whose inputs are absent or incomplete is skipped and recorded, never
    swallowed: the run reports which groups produced no values and why, and exits
    non-zero unless --allow-partial is given. Only the errors a missing or short
    record can raise are caught; anything else is a real defect and propagates.
    """
    try:
        yield
    except (FileNotFoundError, KeyError, IndexError, ValueError, TypeError,
            ZeroDivisionError, SectionSkipped) as exc:
        skipped.append((name, "%s: %s" % (type(exc).__name__, exc)))


def _values_valGmean(lines, data):
    """Device-map geometric means behind the headline gains."""
    DEV = ("berlin_star", "miami_star", "willow_star")
    for spec_, nm_ in (("surf2d:5", "SurfFive"), ("color2d:5", "ColorFive")):
        g_ = np.array([x for x in ((r.get("gain_xfit") if r.get("gain_xfit") is not None
                                    else r.get("gain_bb"))
                                   for r in data if r["spec"] == spec_
                                   and r["noise"] in DEV and r["tier"] == "dist") if x])
        if len(g_) >= 100:
            lines.append("\\newcommand{\\valGmean%sDevice}{%s}" % (nm_, fmt_pct(float(np.exp(np.log(g_).mean())))))
            lines.append("\\newcommand{\\valUpto%sDevice}{%s}" % (nm_, fmt_pct(float(g_.max()))))
            lines.append("\\newcommand{\\valNDeviceMaps%s}{%d}" % (nm_, len(g_)))
    import json as _json2
    _bb = _json2.load(open(_rp(_CFG.study_dir + "/typeA_p01_data.json")))
    _cells = {}
    for r in _bb:
        if r["spec"].startswith("BB") and r["noise"] in DEV:
            _cells.setdefault((r["spec"], r["noise"], r["fseed"]), {})[r["frame_name"]] = r
    _gs = []
    for _k, _m in _cells.items():
        if len(_m) < 5 or any(min(v["ax"], v["az"]) < 500 for v in _m.values()):
            continue
        _wa = {n: max(v["ax"] / v["Nx"], v["az"] / v["Nz"]) for n, v in _m.items()}
        _gs.append(min(_wa[n] for n in BASELINES) / _wa["Cham"])
    if len(_gs) >= 10:
        _gs = np.array(_gs)
        lines.append("\\newcommand{\\valGmeanBBDevice}{%s}" % fmt_pct(float(np.exp(np.log(_gs).mean()))))
        lines.append("\\newcommand{\\valUptoBBDevice}{%s}" % fmt_pct(float(_gs.max())))

def _values_valAbst(lines, data):
    """Deployment guardrail: abstention rates and what they cost."""
    ab = json.load(open(rpath(_CFG.out_dir + "/abstention.json")))
    for spec, nm in (("surf2d:5", "Surf"), ("color2d:5", "Color")):
        rs = [r for r in ab if r["spec"] == spec and r.get("margin") and r.get("gain_xfit")]
        g = np.array([r["gain_xfit"] for r in rs]); mg = np.array([r["margin"] for r in rs])
        keep = mg >= 1.1
        lines.append("\\newcommand{\\valAbst%sN}{%d}" % (nm, len(rs)))
        lines.append("\\newcommand{\\valAbst%sCorr}{%.2f}" % (nm, np.corrcoef(np.log(mg), np.log(g))[0, 1]))
        lines.append("\\newcommand{\\valAbst%sAbstPct}{%d}" % (nm, round(100 * (1 - keep.mean()))))
        lines.append("\\newcommand{\\valAbst%sLossAll}{%d}" % (nm, int((g < 1).sum())))
        lines.append("\\newcommand{\\valAbst%sLossKept}{%d}" % (nm, int((g[keep] < 1).sum())))
        lines.append("\\newcommand{\\valAbst%sMedAll}{%.2f}" % (nm, float(np.median(g))))
        lines.append("\\newcommand{\\valAbst%sMedKept}{%.2f}" % (nm, float(np.median(g[keep]))))

def _values_valCaseGammaCss(lines, data):
    """Distance-3 case study: surrogate terms per policy."""
    cs = _json.load(open(_rp(_CFG.out_dir + "/casestudy_d3.json")))
    g_css, g_ch = cs["mech_table"][0]
    def _sci10(v):
        e = int(np.floor(np.log10(v)))
        return "$%.1f{\\times}10^{%d}$" % (v / 10 ** e, e)
    lines.append("\\newcommand{\\valCaseGammaCss}{%s}" % _sci10(g_css))
    lines.append("\\newcommand{\\valCaseGammaCham}{%s}" % _sci10(g_ch))
    ua = cs.get("u_axes")
    if ua:   # Step-4 per-axis totals (vertical X/Z panel, 2026-07-19)
        lines.append("\\newcommand{\\valCaseUxDrop}{%s}" % fmt(ua["CSS"]["X"] / ua["Cham"]["X"]))
        lines.append("\\newcommand{\\valCaseUzDrop}{%s}" % fmt(ua["CSS"]["Z"] / ua["Cham"]["Z"]))

def _values_valRebuildGeoLo(lines, data):
    """DEM and decoder rebuild cost."""
    rb = _json.load(open(_rp(_CFG.out_dir + "/rebuild_timing.json")))
    geo = [r["rebuild_s"] for r in rb
           if not r["spec"].startswith("BB") and r["spec"] != "color2d:7"]  # color d7 dropped 2026-07-19
    bbb = [r["rebuild_s"] for r in rb if r["spec"].startswith("BB")]
    if geo and bbb:
        lines.append("\\newcommand{\\valRebuildGeoLo}{%s}" % fmt(min(geo)))
        lines.append("\\newcommand{\\valRebuildGeoHi}{%s}" % fmt(max(geo)))
        lines.append("\\newcommand{\\valRebuildBBLo}{%s}" % fmt(min(bbb)))
        lines.append("\\newcommand{\\valRebuildBBHi}{%s}" % fmt(max(bbb)))

def _values_valRestartSpreadGeoPct(lines, data):
    """Spread across cold CEM restarts."""
    rp = _json.load(open(_rp(_CFG.out_dir + "/restart_probe.json")))
    geo = [r for r in rp if not r["spec"].startswith("BB")]
    bbr = [r for r in rp if r["spec"].startswith("BB")]
    if geo:
        lines.append("\\newcommand{\\valRestartSpreadGeoPct}{%s}" % fmt(100 * max(r["final_spread"] - 1 for r in geo)))
        lines.append("\\newcommand{\\valRestartIter}{%d}" % max(r["hit_iter_max"] for r in geo))
    if bbr:
        lines.append("\\newcommand{\\valRestartSpreadBBPct}{%d}" % int(100 * max(r["final_spread"] - 1 for r in bbr) + 0.5))

def _values_valRandGapLoPct(lines, data):
    """Reach of random frames against the selected one."""
    rr = _json.load(open(_rp(_CFG.out_dir + "/randreach_s3.json")))
    gaps = [100 * (r["u_rand_min"] / r["u_cem"] - 1) for r in rr if r["noise"] == "willow_star"]
    xyz = [r["u_rand_min"] / r["u_cem"] for r in rr if r["noise"].startswith("xyz")]
    if gaps:
        lines.append("\\newcommand{\\valRandGapLoPct}{%s}" % fmt(min(gaps)))
        lines.append("\\newcommand{\\valRandGapHiPct}{%d}" % int(max(gaps) + 0.5))
    if xyz:
        lines.append("\\newcommand{\\valRandGapXyz}{%s}" % fmt_pct(max(xyz)))

def _values_valReweightMed(lines, data):
    """Decoding headroom left after reweighting."""
    rw = _json.load(open(_rp(_CFG.out_dir + "/reweight_headroom.json")))
    ratios = [r["ler_blind"] / r["ler_weighted"] for r in rw
              if r.get("ler_weighted") and r.get("ler_blind")]
    if ratios:
        lines.append("\\newcommand{\\valReweightMed}{%s}" % fmt_pct(np.median(ratios)))
        lines.append("\\newcommand{\\valReweightLo}{%s}" % fmt_pct(min(ratios)))
        lines.append("\\newcommand{\\valReweightHi}{%s}" % fmt_pct(max(ratios)))

def _values_valDriftMedRetention(lines, data):
    """Gain retained when the deployed map drifts from the calibrated one."""
    dr = _json.load(open(_rp(_CFG.out_dir + "/drift_study.json")))
    for r in dr: _recompute_gains(r)
    nom = {(r["spec"], r["base"], r["fseed"]): headline_gain(r) for r in dr if r["sigma"] == 0.0}
    rets = [headline_gain(r) / nom[(r["spec"], r["base"], r["fseed"])]
            for r in dr if r["sigma"] == 0.5
            and nom.get((r["spec"], r["base"], r["fseed"])) and headline_gain(r)]
    if rets:
        lines.append("\\newcommand{\\valDriftMedRetention}{%s}" % fmt(np.median(rets)))
        lines.append("\\newcommand{\\valDriftLo}{%s}" % fmt(min(rets)))
        lines.append("\\newcommand{\\valDriftHi}{%s}" % fmt(max(rets)))

def _values_valSixframeGainXyz(lines, data):
    """Binary vs full 6^n frame space, and the per-code frame statistics."""
    ab = _json.load(open(_rp(_CFG.out_dir + "/sixframe_typea.json")))
    BASE = BASELINES
    mx6 = mxT = mx62 = 0.0
    panels = {}
    cells = {}
    SIXFRAME_SPECS = ("surf2d:5", "color2d:5")   # figure scope (user 2026-07-19: surf+color only)
    # field scope pinned to the figure's groups: the eta campaign folded xyz:20/50/100
    # into the same master, and an unscoped aggregate would quote gains the figure
    # does not plot (guard caught +271 -> +1525 drift, 2026-07-19)
    SIXFRAME_FIELDS = ("berlin_star", "miami_star", "willow_star", "xyz:10", "xyz:50")
    for r in ab:
        if r["spec"] not in SIXFRAME_SPECS or r["noise"] not in SIXFRAME_FIELDS:
            continue
        cells.setdefault((r["spec"], r["noise"], r["fseed"]), {})[r["frame_name"]] = r["lerA"]
    for (spec, noise, fseed), m in cells.items():
        if not all(k in m for k in BASE):
            continue
        bbl = min(m[k] for k in BASE)
        for arm in ("Cham2", "Cham6"):
            if arm not in m or m[arm] <= 0:
                continue
            panels.setdefault((spec, noise), {}).setdefault(arm, []).append(bbl / m[arm])
        if noise.startswith("xyz") and m.get("Cham6", 0) > 0:
            mx6 = max(mx6, bbl / m["Cham6"])
            mxT = max(mxT, m["Tiurev"] / m["Cham6"])
            if m.get("Cham2", 0) > 0:
                mx62 = max(mx62, m["Cham2"] / m["Cham6"])
    # the "measured maps" claim (sec-VI-B + Fig 7 caption) counts MEASURED panels only,
    # not the synthetic xyz ones (six-frame ahead 6/9 measured; xyz adds 3/3).
    MEASNZ = ("berlin_star", "miami_star", "willow_star")
    mp = {k: v for k, v in panels.items() if k[1] in MEASNZ}
    ahead = sum(1 for v in mp.values()
                if v.get("Cham6") and v.get("Cham2")
                and np.mean(v["Cham6"]) >= np.mean(v["Cham2"]))
    tot = sum(1 for v in mp.values() if v.get("Cham6") and v.get("Cham2"))
    if mx6:
        lines.append("\\newcommand{\\valSixframeGainXyz}{%s}" % fmt_pct(mx6))
        lines.append("\\newcommand{\\valSixframeVsTiurev}{%s}" % fmt_pct(mxT))
        lines.append("\\newcommand{\\valSixframeVsBinary}{%s}" % fmt_pct(mx62))
        lines.append("\\newcommand{\\valSixframePanelsAhead}{%d}" % ahead)
        lines.append("\\newcommand{\\valSixframePanelsTotal}{%d}" % tot)
        # z-statistics (2026-07-19): measured-map panels are statistical ties;
        # report max |z| instead of a win-rate framing. Poisson SE on the
        # worst-axis Type-A event counts.
        import math as _math
        def _wev(rec):
            lx, lz = rec["ax"]/rec["Nx"], rec["az"]/rec["Nz"]
            return rec["ax"] if lx >= lz else rec["az"]
        recs = {}
        for r in ab:
            if r["spec"] in SIXFRAME_SPECS and r["noise"] in SIXFRAME_FIELDS:
                recs.setdefault((r["spec"], r["noise"], r["fseed"]), {})[r["frame_name"]] = r
        pz = {}
        for (spec, noise), _ in panels.items():
            ds, vs = [], []
            for fs in (0, 1, 2):
                m = recs.get((spec, noise, fs))
                if not m or not all(k in m for k in BASE + ("Cham2", "Cham6")):
                    continue
                bbk = min(BASE, key=lambda k: m[k]["lerA"]); bb = m[bbk]["lerA"]
                g2 = bb/m["Cham2"]["lerA"]; g6 = bb/m["Cham6"]["lerA"]
                e_bb, e2, e6 = _wev(m[bbk]), _wev(m["Cham2"]), _wev(m["Cham6"])
                ds.append(g6 - g2)
                vs.append(g2*g2*(1/e_bb+1/e2) + g6*g6*(1/e_bb+1/e6))
            if ds:
                se = _math.sqrt(sum(vs)/len(vs)/len(ds))
                pz[(spec, noise)] = sum(ds)/len(ds)/se
        meas_z = [abs(z) for (sp, nz), z in pz.items() if nz in MEASNZ]
        xyz_z = [z for (sp, nz), z in pz.items() if nz.startswith("xyz")]
        if meas_z:
            lines.append("\\newcommand{\\valSixframeMeasMaxAbsZ}{%.1f}" % max(meas_z))
        if xyz_z:
            lines.append("\\newcommand{\\valSixframeXyzMaxZ}{%.0f}" % max(xyz_z))
        # per-spec eta=50 arms for the S V-B prose (Fig 8 callout convention:
        # mean over fseeds of bestprior/arm; six-over-binary = ratio of the means)
        def _mp(spec, nz, arm):
            v = panels.get((spec, nz), {}).get(arm)
            return float(np.mean(v)) if v else None
        g2c, g6c = _mp("color2d:5", "xyz:50", "Cham2"), _mp("color2d:5", "xyz:50", "Cham6")
        if g2c and g6c:
            lines.append("\\newcommand{\\valSixframeColorBinFifty}{%s}" % fmt_pct(g2c))
            lines.append("\\newcommand{\\valSixframeColorSixFifty}{%s}" % fmt_pct(g6c))
            lines.append("\\newcommand{\\valSixframeColorSixOverBinFifty}{%s}" % fmt_pct(g6c / g2c))

def _values_valQratio(lines, data):
    """Measurement-to-data error-ratio sweep."""
    qs = _json.load(open(_rp(_CFG.out_dir + "/qratio_sweep.json")))
    for r in qs: _recompute_gains(r)
    anchors = [r for r in data if r["tier"] == "anchor" and r["fseed"] in (0, 1, 2)
               and r["noise"] in ("berlin_star", "miami_star", "willow_star", "xyz:10")
               and r.get("p", P_ANCHOR) == P_ANCHOR]
    NAMES = {"surf2d:5": "Surf", "color2d:5": "Color", "BB72": "BB"}
    floor = []
    for spec, nm in NAMES.items():
        pool = {}
        for r in qs:
            if r["spec"] == spec:
                g = headline_gain(r)
                if g: pool.setdefault(r["q_ratio"], []).append(g)
        g1 = _hg([r for r in anchors if r["spec"] == spec])
        if g1: pool[1.0] = g1
        meds = {q: np.median(v) for q, v in pool.items() if v}
        if 0.0 in meds:
            lines.append("\\newcommand{\\valQratio%sQzero}{%s}" % (nm, fmt(meds[0.0])))
        if 1.0 in meds:
            lines.append("\\newcommand{\\valQratio%sQone}{%s}" % (nm, fmt(meds[1.0])))
        floor += list(meds.values())
    if floor:
        lines.append("\\newcommand{\\valQratioFloor}{%s}" % fmt(min(floor)))

def _values_valPatchWinLoPct(lines, data):
    """Device-patch study: win fraction and per-device medians."""
    from chameleon.estimators import xfit_gain
    pt = _json.load(open(_rp(_CFG.out_dir + "/patch_pheno.json")))
    for r in pt:
        r["gain_xfit"] = xfit_gain(r["masks"], _XFIT_CHUNK)
    cells = {}
    for r in pt:
        g = headline_gain(r)
        if g: cells.setdefault((r["dev"], r["spec"]), []).append(g)
    wins = [100 * np.mean(np.array(v) > 1) for v in cells.values()]
    bydev = {}
    for (dev, sp), v in cells.items():
        bydev.setdefault(dev, []).extend(v)
    meds = [np.median(v) for v in bydev.values()]     # per-device pooled (prose semantics)
    allg = [g for v in cells.values() for g in v]
    lines.append("\\newcommand{\\valPatchWinLoPct}{%d}" % int(min(wins) + 0.5))
    lines.append("\\newcommand{\\valPatchWinHiPct}{%d}" % int(max(wins) + 0.5))
    lines.append("\\newcommand{\\valPatchMedLo}{%s}" % fmt_pct(min(meds)))
    lines.append("\\newcommand{\\valPatchMedHi}{%s}" % fmt_pct(max(meds)))
    lines.append("\\newcommand{\\valPatchMaxGain}{%s}" % fmt_pct(max(allg)))
    lines.append("\\newcommand{\\valPatchN}{%d}" % len(pt))

def _values_valPsweep(lines, data):
    """Physical-rate sweep: gain at the low and high ends of the p grid."""
    ps = _json.load(open(_rp(_CFG.out_dir + "/xyz_psweep_ext.json")))
    for r in ps: _recompute_gains(r)
    for tag, p in (("LoP", 1e-4), ("HiP", 1e-2)):
        v = _hg([r for r in ps if r["p"] == p])
        if v: lines.append("\\newcommand{\\valPsweepMed%s}{%s}" % (tag, fmt_pct(np.median(v))))
    meds = []
    for p in sorted({r["p"] for r in ps}):
        v = _hg([r for r in ps if r["p"] == p and r["spec"] == "surf2d:3"])
        if v: meds.append(np.median(v))
    if meds:
        lines.append("\\newcommand{\\valPsweepSurfThreeLo}{%s}" % fmt_pct(min(meds)))
        lines.append("\\newcommand{\\valPsweepSurfThreeHi}{%s}" % fmt_pct(max(meds)))
    for spec, nm in (("surf2d:5", "SurfFive"), ("color2d:5", "ColorFive")):
        # after the 2026-07-19 shot boost p=5e-4 fully resolves for both d5
        # families; p=1e-4 remains below the MC shot floor (Fig 9 spans 5e-4..1e-2).
        v = _hg([r for r in ps if r["p"] == 5e-4 and r["spec"] == spec])
        if v: lines.append("\\newcommand{\\valPsweep%sLoP}{%s}" % (nm, fmt_pct(np.median(v))))
        v = _hg([r for r in ps if r["p"] == 1e-2 and r["spec"] == spec])
        if v: lines.append("\\newcommand{\\valPsweep%sHiP}{%s}" % (nm, fmt_pct(np.median(v))))

def _values_per_cell_macros(lines, data):
    """Per (code, noise) macros at the anchor and distribution tiers."""
    for spec, S in SPEC.items():
        for field, F in FIELD.items():
            anchor = resolved([r for r in data if r["spec"] == spec and r["noise"] == field
                               and r["tier"] == "anchor" and r.get("p", P_ANCHOR) == P_ANCHOR])
            dist = resolved([r for r in data if r["spec"] == spec and r["noise"] == field
                             and r["tier"] in ("anchor", "dist") and r.get("p", P_ANCHOR) == P_ANCHOR])
            base = "\\val%s%s" % (S, F)
            if anchor:
                x = [r["gain_xfit"] for r in anchor if r.get("gain_xfit")]
                # surf d7 sits at the Monte-Carlo floor where the cross-fit is
                # noise-unstable (swings 4.85<->5.20 between runs) while the plug-in
                # is cfg-stable (~4.4); report the plug-in there (user 2026-07-14).
                if spec == "surf2d:7":
                    g = [r["gain_bb"] for r in anchor if r.get("gain_bb")]
                else:
                    # per-record headline (xfit else plug-in) so a partially-xfit anchor
                    # set never silently averages a subset (review MINOR-1, 2026-07-20)
                    g = [(r["gain_xfit"] if r.get("gain_xfit") is not None else r["gain_bb"])
                         for r in anchor]
                lines.append("\\newcommand{%sGain}{%s}" % (base, fmt_pct(np.mean(g))))
                lines.append("\\newcommand{%sGainLo}{%s}" % (base, fmt_pct(min(g))))
                lines.append("\\newcommand{%sGainHi}{%s}" % (base, fmt_pct(max(g))))
                if x:
                    lines.append("\\newcommand{%sGainXfit}{%s}" % (base, fmt_pct(np.mean(x))))
                ler = np.mean([r["masks"]["Cham"]["ler"] for r in anchor])
                e = int(np.floor(np.log10(ler))); m = ler / 10 ** e
                lines.append("\\newcommand{%sLer}{$%s{\\times}10^{%d}$}" % (base, fmt(m, 1), e))
            else:
                for suf in ("Gain", "GainLo", "GainHi", "GainXfit", "Ler"):
                    lines.append("\\newcommand{%s%s}{\\valTBD}" % (base, suf))
            if len(dist) >= 50:
                g = np.array([(r["gain_xfit"] if r.get("gain_xfit") is not None else r["gain_bb"])
                              for r in dist])
                lines.append("\\newcommand{%sMedian}{%s}" % (base, fmt_pct(np.median(g))))
                lines.append("\\newcommand{%sPFive}{%s}" % (base, fmt_pct(np.percentile(g, 5))))
                lines.append("\\newcommand{%sPNinetyFive}{%s}" % (base, fmt_pct(np.percentile(g, 95))))
                lines.append("\\newcommand{%sLossPct}{%d}" % (base, round(100 * (g < 1 - 1e-9).mean())))
                # significance-filtered regression: the raw LossPct at the 30-event dist floor overcounts
                # near break-even (per-map gain rel-std ~26%). SigLossPct = % of maps whose 95%-one-sided
                # gain CI (from per-map binomial event counts) is still <1 -- i.e. confidently regressing.
                _sn = _ss = 0
                for _r in dist:
                    _gx = _r.get("gain_xfit") if _r.get("gain_xfit") is not None else _r.get("gain_bb")
                    _mm = _r.get("masks", {})
                    if not _gx or "Cham" not in _mm:
                        continue
                    _evc = _mm["Cham"].get("ev", 0)
                    _evb = [(v.get("ler", 9), v.get("ev", 1)) for k, v in _mm.items()
                            if k != "Cham" and v.get("ler") and not v.get("unresolved")]
                    if not _evb or _evc < 1:
                        continue
                    _rel = (1.0 / max(_evc, 1) + 1.0 / max(min(_evb)[1], 1)) ** 0.5
                    _sn += 1
                    if _gx * (1 + 1.645 * _rel) < 1:
                        _ss += 1
                lines.append("\\newcommand{%sSigLossPct}{%d}" % (base, round(100 * _ss / _sn) if _sn else 0))
                lines.append("\\newcommand{%sNMaps}{%d}" % (base, len(g)))
                gc = [r["gain_css"] for r in dist if r.get("gain_css")]
                lines.append("\\newcommand{%sCssMedian}{%s}" % (base, fmt(np.median(gc)))
                             if len(gc) >= 50 else
                             "\\newcommand{%sCssMedian}{\\valTBD}" % base)
            else:
                for suf in ("Median", "PFive", "PNinetyFive", "LossPct", "SigLossPct", "NMaps", "CssMedian"):
                    lines.append("\\newcommand{%s%s}{\\valTBD}" % (base, suf))


def sci(v, nd=1):
    """Scientific-notation LaTeX for a rate, e.g. $1.4{\\times}10^{-3}$."""
    e = int(np.floor(np.log10(v)))
    return "$%s{\\times}10^{%d}$" % (fmt(v / 10 ** e, nd), e)


def _values_case_study_d3(lines):
    """d=3 case-study macros (results/protocol_v1/casestudy_d3.json)."""
    # d=3 case-study macros (results/protocol_v1/casestudy_d3.json)
    cs = rpath(_CFG.out_dir + "/casestudy_d3.json")
    if os.path.exists(cs):
        D = json.load(open(cs))
        for tag, masks in (("", D["masks"]), ("Qzero", D["masks_q0"])):
            bb = min(v["ler"] for k, v in masks.items() if k != "Cham")
            ch = masks["Cham"]["ler"]
            lines.append("\\newcommand{\\valCase%sGain}{%s}" % (tag, fmt_pct(bb / ch)))
            lines.append("\\newcommand{\\valCase%sLerCham}{%s}" % (tag, sci(ch)))
            lines.append("\\newcommand{\\valCase%sLerBB}{%s}" % (tag, sci(bb)))
    else:
        for tag in ("", "Qzero"):
            for suf in ("Gain", "LerCham", "LerBB"):
                lines.append("\\newcommand{\\valCase%s%s}{\\valTBD}" % (tag, suf))


def _values_overview_sixpolicy(lines):
    """Six-policy overview macros, including the CDSC comparison."""
    # six-policy overview macros
    ov = rpath(_CFG.out_dir + "/overview_sixpolicy.json")
    if os.path.exists(ov):
        D = json.load(open(ov))
        name = {"CSS":"Css","XZZX":"Xzzx","ZXXZ":"Zxxz","Tiurev":"Tiurev","CDSC":"Cdsc","Cham":"Cham"}
        for k, v_ in D["masks"].items():
            lines.append("\\newcommand{\\valOv%s}{%s}" % (name[k], sci(v_["ler"], 2)))
        bb = min(v_["ler"] for k, v_ in D["masks"].items() if k not in ("Cham", "CDSC"))
        lines.append("\\newcommand{\\valOvGain}{%s}" % fmt_pct(bb / D["masks"]["Cham"]["ler"]))
        if "CDSC" in D["masks"]:
            lines.append("\\newcommand{\\valOvCdscGain}{%s}" % fmt_pct(
                D["masks"]["CDSC"]["ler"] / D["masks"]["Cham"]["ler"]))
    # decode-validation (S6.4) + magic-prose (S6.6) macros -- recomputed from raw masks


def _values_decode_validation(lines):
    """Decode-validation and magic-prose macros, recomputed from raw masks."""
    dv = rpath(_CFG.out_dir + "/decode_validation.json")
    if os.path.exists(dv):
        drecs = json.load(open(dv))
        rat, bb = [], []
        for r_ in drecs:
            m_ = r_["masks"]
            if m_["Cham"].get("unresolved") or m_["ChamMC"].get("unresolved"):
                continue
            q_ = m_["ChamMC"]["ler"] / m_["Cham"]["ler"]
            rat.append(q_)
            if r_["spec"] == "BB72":
                bb.append(q_)
        if rat:
            ra = np.array(rat)
            lines.append("\\newcommand{\\valDecvalN}{%d}" % len(ra))
            lines.append("\\newcommand{\\valDecvalCells}{%d}" % len({(r_["spec"], r_["noise"]) for r_ in drecs}))
            lines.append("\\newcommand{\\valDecvalW}{%d}" % int((ra < 0.9).sum()))
            lines.append("\\newcommand{\\valDecvalT}{%d}" % int(((ra >= 0.9) & (ra <= 1.1)).sum()))
            lines.append("\\newcommand{\\valDecvalL}{%d}" % int((ra > 1.1).sum()))
            lines.append("\\newcommand{\\valDecvalPooled}{%.2f}" % np.median(ra))
        if bb:
            ba = np.array(bb)
            lines.append("\\newcommand{\\valDecvalBBW}{%d}" % int((ba < 0.9).sum()))
            lines.append("\\newcommand{\\valDecvalBBT}{%d}" % int(((ba >= 0.9) & (ba <= 1.1)).sum()))
            lines.append("\\newcommand{\\valDecvalBBL}{%d}" % int((ba > 1.1).sum()))
            lines.append("\\newcommand{\\valDecvalBBMed}{%.2f}" % np.median(ba))
    # magic-state consumer (S6.magic): single-axis vs worst-axis objective, Willow only
    # (single vs worst, not single vs CSS -- the magic-state table /
    # make_magic_table.py is the source of truth for the table; these macros feed the prose).
    # n=10 willow campaign (2026-07-24, >=500-event floor) -- same source as
    # make_magic_table.py, superseding the 3-seed ftops_magic_pheno.json batch


def _values_magic_willow(lines):
    """The n=10 Willow worst-arm campaign macros."""
    mg = rpath(_CFG.study_dir + "/magic_worst_arm_willow_n10.json")
    if os.path.exists(mg):
        mrecs = json.load(open(mg))
        willow = [r_ for r_ in mrecs if not r_.get("unres")]
        MAGSPEC = {"surf2d:5": "SurfFive", "color2d:5": "ColorFive", "BB72": "BBSeventyTwo"}
        for spec, nm in MAGSPEC.items():
            rs = [r_ for r_ in willow if r_["spec"] == spec]
            if len(rs) < 3:
                continue
            gains = [r_["gain_vs_worst"] for r_ in rs]
            lines.append("\\newcommand{\\valMagic%sWillowGainMean}{%s}" % (nm, fmt_pct(np.mean(gains))))
            lines.append("\\newcommand{\\valMagic%sWillowGainLo}{%s}" % (nm, fmt_pct(min(gains))))
            lines.append("\\newcommand{\\valMagic%sWillowGainHi}{%s}" % (nm, fmt_pct(max(gains))))
    # BB guardrailed-deployment macros (bb_guardrail_full.json; emit only 5-seed-complete groups)


def _values_bb_guardrail(lines):
    """BB guardrailed-deployment macros; only 5-seed-complete groups are emitted."""
    gj = rpath(_CFG.out_dir + "/bb_guardrail_full.json")
    if os.path.exists(gj):
        grecs = json.load(open(gj))
        from collections import defaultdict as _dd
        groups = _dd(list)
        for r_ in grecs:
            groups[(r_["spec"], r_["noise"])].append(r_["guardrailed_gain"])
        gname = {"BB18": "BBEighteen", "BB36": "BBThirtySix", "BB72": "BBSeventyTwo"}
        fname = {"berlin_star": "Berlin", "miami_star": "Miami",
                 "willow_star": "Willow", "xyz:10": "Xyz"}
        for (sp_, nz_), gs_ in sorted(groups.items()):
            if len(gs_) < 5 or sp_ not in gname:
                continue
            base_ = "\\val%sGuard%s" % (gname[sp_], fname[nz_])
            lines.append("\\newcommand{%sMean}{%s}" % (base_, fmt(float(np.mean(gs_)))))
            lines.append("\\newcommand{%sLo}{%s}" % (base_, fmt(min(gs_))))
            lines.append("\\newcommand{%sHi}{%s}" % (base_, fmt(max(gs_))))
    # decode-cost macros (tab:decodecost + SS3 quote): measured BB s/shot + BB108 anchor LER


def _values_decode_cost(lines):
    """Decode-cost macros: measured BB seconds/shot and the BB108 anchor LER."""
    tj = rpath(_CFG.out_dir + "/bb_shot_timing.json")
    if os.path.exists(tj):
        T = {r["spec"]: r["s_per_shot"] for r in json.load(open(tj))
             if r["spec"] in ("BB18", "BB36", "BB72")}
        from chameleon.config import SelectorConfig
        _sc = SelectorConfig()
        budget = _sc.cem_restarts * _sc.cem_iters * _sc.cem_pop
        exps = [np.log10(sv / 1e-6) for sv in T.values()]
        lines.append("\\newcommand{\\valDecShotMsLo}{%.1f}" % (min(T.values()) * 1e3))
        lines.append("\\newcommand{\\valDecShotMsHi}{%.1f}" % (max(T.values()) * 1e3))
        lines.append("\\newcommand{\\valDecSlowLoExp}{%.1f}" % min(exps))
        lines.append("\\newcommand{\\valDecSlowHiExp}{%.1f}" % max(exps))
        if "BB72" in T:
            lines.append("\\newcommand{\\valDecSlowBBSeventyTwoExp}{%.1f}" % np.log10(T["BB72"] / 1e-6))
            lines.append("\\newcommand{\\valDecSlowBBSeventyTwoK}{%.1f}" % (T["BB72"] / 1e-6 / 1000))
        bbx = rpath(_CFG.out_dir + "/protocol_v1_B_bbext.json")
        if os.path.exists(bbx):
            rs = [r for r in json.load(open(bbx))
                  if r["spec"] == "BB72" and r["noise"] == "berlin_star"
                  and r.get("masks", {}).get("Cham", {}).get("ler")]
            if rs:
                ler = np.mean([r["masks"]["Cham"]["ler"] for r in rs])
                lines.append("\\newcommand{\\valDecBBSeventyTwoDays}{%.1f}"
                             % (budget * (100.0 / ler) * 1e-6 / 86400.0))


def _derive_cross_cell_macros(lines):
    """Macros derived from the freshly emitted lines, never from storage."""
    _nine, _nmaps = {}, {}
    for m, value in lines.matching(r"valColor(Three|Five|Seven)(Berlin|Miami|Willow)Median"):
        _nine[(m.group(1), m.group(2))] = float(value)
    for m, value in lines.matching(r"valColor(Three|Five|Seven)(Berlin|Miami|Willow)NMaps"):
        _nmaps[(m.group(1), m.group(2))] = int(value)
    if len(_nine) == 9:
        # values are already percent-change (fmt_pct upstream); preserve the sign, no re-transform
        lines.append("\\newcommand{\\valColorMedNineLo}{%+d}" % round(min(_nine.values())))
        lines.append("\\newcommand{\\valColorMedNineHi}{%+d}" % round(max(_nine.values())))
    if len(_nmaps) == 9:
        lines.append("\\newcommand{\\valColorNMapsLo}{%d}" % min(_nmaps.values()))
    # required-artifact gate: a MISSING or malformed side-artifact is swallowed by the
    # except-pass / os.path.exists blocks above and silently DROPS its whole macro group,
    # which then renders "??" in the compiled PDF. Check one representative USED macro per


def _warn_missing_macro_groups(lines, skipped):
    """Warn by name when a required macro group is absent from the output."""
    # required group; warn by NAME + source artifact if absent. Warning-only.
    _emitted = {"\\" + _n for _n in lines.names()}
    _required = {
        r"\valAbstSurfCorr": "abstention.json (guardrail, S7)",
        r"\valDecvalPooled": "decode_validation.json (winner's curse, S5b)",
        r"\valOvCdscGain": "overview_sixpolicy.json (CDSC head-to-head, S6)",
        r"\valReweightMed": "reweight_headroom.json (decoding headroom, S6)",
        r"\valDriftMedRetention": "drift_study.json (drift stability, S7)",
        r"\valPatchN": "patch_pheno.json (device patch sweep, S6)",
    }
    _missing = [(_m, _src) for _m, _src in _required.items() if _m not in _emitted]
    if _missing:
        print("WARNING make_values: %d REQUIRED macro group(s) absent -> the paper will render '??'. "
              "Regenerate the source artifact:" % len(_missing))
        for _m, _src in _missing:
            print("   %s  <-  %s" % (_m, _src))
    # sanity gate: warn on macros that silently computed garbage (None/nan/inf/empty) --
    # the failure class the graceful except-blocks above cannot catch (a block that
    # succeeds but emits a bad value). Warning-only; does not alter the emitted file.
    # ---------------------------------------------------------------- write
    if lines.suspicious:
        print("WARNING: %d value(s) came out empty or non-finite:" % len(lines.suspicious))
        for name, value in lines.suspicious[:20]:
            print("    %s -> %r" % (name, value))

    if skipped:
        print("WARNING: %d value group(s) could not be computed:" % len(skipped))
        for name, why in skipped:
            print("    %-28s %s" % (name, why))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--allow-partial", action="store_true",
                    help="exit 0 even if some value groups could not be computed")
    args = ap.parse_args()

    data = load()
    skipped = []
    lines = ValueTable()
    lines.append("\\newcommand{\\valTBD}{[TBD]}")
    ncells = len(data)
    lines.append("\\newcommand{\\valCellsDone}{%d}" % ncells)
    # motivation stats: strong-bias tails from the raw device pools (eta_q>1.5)
    # and prior-tailoring loss fractions over all resolved measured-map instances
    from chameleon.fields import pool3
    for dev, nm in (("berlin", "Berlin"), ("miami", "Miami"), ("willow", "Willow")):
        P = np.asarray(pool3(dev))
        rX, rZ = P[:, 0] + P[:, 1], P[:, 2] + P[:, 1]
        eta = np.maximum(rX / np.maximum(rZ, 1e-12), rZ / np.maximum(rX, 1e-12))
        pct = 100 * (eta > 1.5).mean()
        lines.append("\\newcommand{\\valMot%sStrongPct}{%d}" % (nm, int(np.floor(pct + 0.5))))  # half-up: Berlin 12.5 -> 13
        if dev == "willow":
            lines.append("\\newcommand{\\valMotMaxImb}{%.1f}" % (np.ceil(eta.max() * 10) / 10))
    meas = [r for r in data if r["noise"] in ("berlin_star", "miami_star", "willow_star")]
    fr = {}
    for nm in ("XZZX", "ZXXZ", "Tiurev"):
        pairs = [(r["masks"][nm]["ler"], r["masks"]["CSS"]["ler"]) for r in meas
                 if nm in r["masks"] and r["masks"][nm].get("ler") and r["masks"]["CSS"].get("ler")
                 and not r["masks"][nm].get("unresolved") and not r["masks"]["CSS"].get("unresolved")]
        fr[nm] = (sum(a > b for a, b in pairs), len(pairs))
    n_inst = max(v[1] for v in fr.values())
    lines.append("\\newcommand{\\valMotInstances}{%s}" % "{:,}".format(n_inst).replace(",", "{,}"))
    lines.append("\\newcommand{\\valMotLossFixedPct}{%d}" % round(
        100 * max(fr["XZZX"][0] / fr["XZZX"][1], fr["ZXXZ"][0] / fr["ZXXZ"][1])))
    lines.append("\\newcommand{\\valMotLossTiurevPct}{%d}" % round(100 * fr["Tiurev"][0] / fr["Tiurev"][1]))

    # Block A/B (R64): sweep + patch macros, headline estimator (xfit-first)
    # eta-tier (from the already-loaded protocol data)
    eta = [r for r in data if r.get("tier") == "eta"]
    def _etacell(spec, e):
        return _hg([r for r in eta if r["spec"] == spec and r["noise"] == "xyz:%d" % e])
    v = _etacell("surf2d:5", 100)
    if v: lines.append("\\newcommand{\\valEtaSurfFiveHundMed}{%s}" % fmt_pct(np.median(v)))
    v = _etacell("color2d:5", 100)
    if v: lines.append("\\newcommand{\\valEtaColorFiveHundMed}{%s}" % fmt_pct(np.median(v)))
    v = _etacell("color2d:7", 50)
    if v: lines.append("\\newcommand{\\valEtaColorSevenFiftyMed}{%s}" % fmt_pct(np.median(v)))
    v = _etacell("color2d:7", 100)
    if v:
        lines.append("\\newcommand{\\valEtaColorSevenHundMed}{%s}" % fmt_pct(np.median(v)))
        lines.append("\\newcommand{\\valEtaColorSevenHundLo}{%s}" % fmt_pct(min(v)))
        lines.append("\\newcommand{\\valEtaColorSevenHundHi}{%s}" % fmt_pct(max(v)))
    v = _etacell("BB72", 20)   # BB72 eta-sweep peak (its highest resolved eta point)
    if v: lines.append("\\newcommand{\\valEtaBBSeventyTwoTwentyMed}{%s}" % fmt_pct(np.median(v)))
    # p-sweep side artifact
    with section("valPsweep", skipped):
        _values_valPsweep(lines, data)
    # patch study
    with section("valPatchWinLoPct", skipped):
        _values_valPatchWinLoPct(lines, data)

    # Block C: BB72 anchor spread (measured fields)
    bb = [r for r in data if r["spec"] == "BB72" and r["tier"] == "anchor"
          and r["noise"] in ("berlin_star", "miami_star", "willow_star")
          and r.get("p", P_ANCHOR) == P_ANCHOR]
    v = _hg(bb)
    if v:
        lines.append("\\newcommand{\\valBBSeventyTwoAnchorLo}{%s}" % fmt_pct(min(v)))
        lines.append("\\newcommand{\\valBBSeventyTwoAnchorHi}{%s}" % fmt_pct(max(v)))
        fm = []
        for nz in ("berlin_star", "miami_star", "willow_star"):
            fv = _hg([r for r in bb if r["noise"] == nz])
            if fv: fm.append(np.mean(fv))
        if fm:
            lines.append("\\newcommand{\\valBBSeventyTwoFieldMeanLo}{%s}" % fmt_pct(min(fm)))
            lines.append("\\newcommand{\\valBBSeventyTwoFieldMeanHi}{%s}" % fmt_pct(max(fm)))
    # Block D: qratio per-family pooled medians (q=1 from main-grid anchors)
    with section("valQratio", skipped):
        _values_valQratio(lines, data)
    # Block E: six-frame ablation, TYPE-A basis (2026-07-17: switched from total-LER
    # binary_ablation.json to sixframe_typea.json, matching Table V's convention; BB72/BB36
    # excluded from this block until the BB36 Type-A campaign lands -- surf2d:5/color2d:5 only)
    with section("valSixframeGainXyz", skipped):
        _values_valSixframeGainXyz(lines, data)

    # Block G/H: probe + side-study macros (frozen artifacts, single source of truth)
    with section("valDriftMedRetention", skipped):
        _values_valDriftMedRetention(lines, data)
    with section("valReweightMed", skipped):
        _values_valReweightMed(lines, data)
    with section("valRandGapLoPct", skipped):
        _values_valRandGapLoPct(lines, data)
    with section("valRestartSpreadGeoPct", skipped):
        _values_valRestartSpreadGeoPct(lines, data)
    with section("valRebuildGeoLo", skipped):
        _values_valRebuildGeoLo(lines, data)
    with section("valCaseGammaCss", skipped):
        _values_valCaseGammaCss(lines, data)
    # abstention guardrail stats (SS7) from the side artifact
    with section("valAbst", skipped):
        _values_valAbst(lines, data)
    _values_per_cell_macros(lines, data)
    # Device-map geometric means (geomean + up-to headline,
    # surface/color from the dist-tier device maps; BB from RESOLVED p=0.01 device cells
    # only, so WIP cells never leak into the abstract and the number self-updates on fold)
    with section("valGmean", skipped):
        _values_valGmean(lines, data)
    _values_case_study_d3(lines)
    _values_overview_sixpolicy(lines)
    _values_decode_validation(lines)
    _values_magic_willow(lines)
    _values_bb_guardrail(lines)
    _values_decode_cost(lines)
    # derived cross-cell macros (recomputed from the freshly emitted lines, never stored)
    _derive_cross_cell_macros(lines)
    out = rpath(OUT_DIR, "data", "paper_values.csv")
    _warn_missing_macro_groups(lines, skipped)
    lines.write_csv(out)
    print("wrote %s (%d values, %d cells, %d group(s) skipped)"
          % (out, len(lines), ncells, len(skipped)))

    if skipped and not args.allow_partial:
        # A silently missing group used to mean a number vanished from the paper
        # with no signal. Fail instead, and let --allow-partial opt out.
        sys.exit("incomplete: %d value group(s) skipped (use --allow-partial to accept)"
                 % len(skipped))


if __name__ == "__main__":
    main()
