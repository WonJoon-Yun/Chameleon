#!/usr/bin/env python3
"""Export every measured result as tabular data.

This is the artifact's primary output. Rather than shipping plotting code, it
emits the numbers behind each claim in a form that can be inspected, diffed and
re-analysed directly:

    output/data/*.csv                one tidy CSV per result family
    output/chameleon_results.xlsx    the same tables, one sheet each

Usage
    python3 data_generator/export_results.py            # CSV + XLSX
    python3 data_generator/export_results.py --csv-only # skip the workbook

Every gain is RECOMPUTED from the stored per-policy masks at export time
(chameleon.records), never read from a serialized field, so the export cannot
carry a stale estimator.
"""
import os, sys, csv, json, argparse

sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "src")))

from chameleon._root import rpath
from chameleon.config import POLICIES as _POLICIES, ProtocolConfig
from chameleon.records import load_protocol, headline_gain, resolved, OUT_DIR

_CFG = ProtocolConfig()
# The codes the deployed configuration measures. The records also hold cells for
# codes that were retired from the matrix (BB30, BB108, color2d:7): real
# measurements, kept as evidence, but not part of what the paper reports. Rows
# carry a `deployed` flag so the two can be told apart without deleting data.
DEPLOYED_CODES = frozenset(_CFG.codes)
PROTO = _CFG.out_dir          # results/protocol_v1
STUDY = _CFG.study_dir        # results/typeA_investigation
DATA_DIR = rpath(OUT_DIR, "data")
POLICIES = list(_POLICIES)


# ---------------------------------------------------------------- helpers
def write_csv(name, rows, header):
    """Write one tidy table; returns (name, rows) for the workbook stage."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name + ".csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print("  %-34s %6d rows  -> %s" % (name, len(rows), os.path.relpath(path, rpath(""))))
    return name, header, rows


def load_json(relpath):
    p = rpath(relpath)
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- exports
def export_protocol_cells():
    """Per-cell LER for every policy, plus the recomputed gains.

    One row per (code, noise map, physical rate, map seed, tier). This single
    table underlies the main results table and the aggregate gain statistics.
    """
    recs = load_protocol()
    header = (["code", "deployed", "noise", "p", "map_seed", "tier", "model", "q_ratio"]
              + ["ler_%s" % k for k in POLICIES]
              + ["events_%s" % k for k in POLICIES]
              + ["shots_%s" % k for k in POLICIES]
              + ["unresolved_%s" % k for k in POLICIES]
              + ["gain_vs_css", "gain_vs_best_baseline", "gain_crossfit", "gain_headline", "cfg_id"])
    rows = []
    for r in sorted(recs, key=lambda r: (r["spec"], r["noise"], r["p"], r["fseed"])):
        m = r.get("masks") or {}
        row = [r["spec"], r["spec"] in DEPLOYED_CODES, r["noise"], r["p"], r["fseed"],
               r.get("tier"), r.get("model"), r.get("q_ratio")]
        for field in ("ler", "ev", "N", "unresolved"):
            row += [(m.get(k) or {}).get(field) for k in POLICIES]
        row += [r.get("gain_css"), r.get("gain_bb"), r.get("gain_xfit"),
                headline_gain(r), r.get("cfg_id")]
        rows.append(row)
    return write_csv("protocol_cells", rows, header)


def export_gain_summary():
    """Per-code aggregate of the headline gain: min / mean / max over cells.

    The paper reports gains as min/max/mean over all measured cells; this table
    is that reduction, computed from protocol_cells. Tiers are kept separate:
    they run different shot budgets and event floors, so pooling them would mix
    precision levels.
    """
    recs = resolved(load_protocol())
    by = {}
    for r in recs:
        g = headline_gain(r)
        if g is None:
            continue
        by.setdefault((r["spec"], r["noise"], r["p"], r.get("tier")), []).append(g)
    rows = []
    for (spec, noise, p, tier), gs in sorted(by.items(), key=lambda kv: tuple(map(str, kv[0]))):
        gs = sorted(gs)
        n = len(gs)
        rows.append([spec, spec in DEPLOYED_CODES, noise, p, tier, n, min(gs), sum(gs) / n,
                     gs[n // 2], max(gs), sum(1 for g in gs if g < 1.0)])
    return write_csv("gain_summary", rows,
                     ["code", "deployed", "noise", "p", "tier", "n_cells", "gain_min", "gain_mean",
                      "gain_median", "gain_max", "n_cells_worse_than_css"])


def export_policy_comparison():
    """Head-to-head LER of every deformation policy on the same cell.

    Long form (one row per cell x policy), which is what a reviewer needs to
    check that no policy was scored on a different noise map than another.

    (code, noise, p, map_seed, tier) is the join key back to protocol_cells: the
    same map is measured at more than one tier, with different shot budgets and
    therefore different rates, so the tier is part of the identity of a cell.
    """
    recs = load_protocol()
    rows = []
    for r in sorted(recs, key=lambda r: (r["spec"], r["noise"], r["p"], r["fseed"])):
        m = r.get("masks") or {}
        for pol in POLICIES:
            d = m.get(pol)
            if not d:
                continue
            rows.append([r["spec"], r["spec"] in DEPLOYED_CODES, r["noise"], r["p"],
                         r["fseed"], r.get("tier"), pol,
                         d.get("ler"), d.get("ev"), d.get("N"),
                         d.get("evX"), d.get("evZ"), bool(d.get("unresolved"))])
    return write_csv("policy_comparison", rows,
                     ["code", "deployed", "noise", "p", "map_seed", "tier", "policy", "ler",
                      "events", "shots", "events_X", "events_Z", "unresolved"])


def export_main_table():
    """The paper's main results table, as data.

    One row per (code, noise field) at the anchor tier: the mean LER of each
    policy over the anchor draws, and the gain reduced the way the paper reports
    it (mean over draws, with the min and max across draws).
    """
    recs = [r for r in load_protocol() if r.get("tier") == "anchor"]
    by = {}
    for r in recs:
        by.setdefault((r["spec"], r["noise"], r["p"]), []).append(r)
    rows = []
    for (spec, noise, p), cells in sorted(by.items()):
        row = [spec, spec in DEPLOYED_CODES, noise, p, len(cells)]
        for pol in POLICIES:
            vals = [(c["masks"].get(pol) or {}).get("ler") for c in cells
                    if c.get("masks") and (c["masks"].get(pol) or {}).get("ler")]
            row.append(sum(vals) / len(vals) if vals else None)
        gs = [g for g in (headline_gain(c) for c in cells) if g is not None]
        row += [len(gs),
                (sum(gs) / len(gs)) if gs else None,
                min(gs) if gs else None,
                max(gs) if gs else None]
        rows.append(row)
    return write_csv("table_main_results", rows,
                     ["code", "deployed", "noise", "p", "n_draws"]
                     + ["mean_ler_%s" % k for k in POLICIES]
                     + ["n_resolved", "gain_mean", "gain_min", "gain_max"])


def read_paper_values():
    """Report the macro table written by data_generator/paper_values.py."""
    path = os.path.join(DATA_DIR, "paper_values.csv")
    if not os.path.exists(path):
        print("  %-34s (run data_generator/paper_values.py first)" % "paper_values")
        return None
    with open(path) as fh:
        rd = list(csv.reader(fh))
    print("  %-34s %6d rows  -> %s" % ("paper_values", len(rd) - 1,
                                       os.path.relpath(path, rpath(""))))
    return "paper_values", rd[0], rd[1:]


def export_simple(name, relpath, doc):
    """Flatten a flat list-of-dicts record file into a CSV with a union header."""
    d = load_json(relpath)
    if d is None:
        print("  %-34s (record absent: %s)" % (name, relpath))
        return None
    recs = d if isinstance(d, list) else d.get("rows") or d.get("records")
    if not isinstance(recs, list) or not recs or not isinstance(recs[0], dict):
        print("  %-34s (not a flat record list; kept as JSON)" % name)
        return None
    header, seen = [], set()
    for r in recs:
        for k in r:
            if k not in seen and not isinstance(r[k], (dict, list)):
                seen.add(k); header.append(k)
    rows = [[r.get(k) for k in header] for r in recs]
    return write_csv(name, rows, header)


def export_enum_convergence():
    """Weight-cutoff convergence of the ambiguity set (one row per cell x cutoff).

    Shows what the cutoff W buys: how many operators are retained, what the LUT
    costs, how long enumeration takes, and how far the frame picked at that
    cutoff is from the one picked at the deepest cutoff.
    """
    rows = []
    for fam_file in ("m2_enum_convergence_geo", "m2_enum_convergence_bb"):
        d = load_json("%s/%s.json" % (STUDY, fam_file))
        if not d:
            continue
        for family, cells in d.items():
            for c in cells or []:
                for dep in c.get("depths", []):
                    reg = dep.get("regret_pct_on_deepest") or []
                    ham = dep.get("hamming_to_deepest") or []
                    rows.append([
                        family, c.get("spec"), c.get("noise"), c.get("p"), c.get("fseed"),
                        c.get("n"), c.get("w_min"), dep.get("extra"), dep.get("wcut"),
                        dep.get("n_X"), dep.get("n_Z"), dep.get("lut_bytes"),
                        dep.get("enum_s"),
                        (sum(dep.get("U_self") or []) / len(dep["U_self"])) if dep.get("U_self") else None,
                        (sum(reg) / len(reg)) if reg else None,
                        max(reg) if reg else None,
                        (sum(ham) / len(ham)) if ham else None,
                        c.get("deployed_nX"), c.get("deployed_nZ"), c.get("U_best_on_deepest"),
                    ])
    if not rows:
        print("  %-34s (records absent)" % "enum_convergence")
        return None
    return write_csv("enum_convergence", rows,
                     ["family", "code", "noise", "p", "map_seed", "n_qubits", "w_min",
                      "extra", "w_cutoff", "n_ops_X", "n_ops_Z", "lut_bytes", "enum_seconds",
                      "U_self_mean", "regret_pct_mean", "regret_pct_max",
                      "hamming_to_deepest_mean", "deployed_n_X", "deployed_n_Z",
                      "U_best_at_deepest"])


SIMPLE = [
    ("search_ablation",      STUDY + "/m3_search_ablation.json",
     "CEM vs greedy / annealing / multi-start at an equal evaluation budget."),
    ("exact_vs_bhattacharyya", STUDY + "/m1_exact_vs_bhatt.json",
     "Exact rare-event likelihood vs the analytic Bhattacharyya bound."),
    ("runtime_breakdown",    STUDY + "/m7_runtime_breakdown.json",
     "Wall-clock breakdown of enumeration, scoring and search."),
    ("drift_robustness",     STUDY + "/m5_drift_robustness.json",
     "Gain retained when the deployed map drifts from the calibrated one."),
    ("decode_validation",    PROTO + "/decode_validation.json",
     "Decode-validated selection vs the decode-free blind pass."),
    ("abstention",           PROTO + "/abstention.json",
     "Deployment guardrail: when Chameleon abstains and what it costs."),
    ("drift_study",          PROTO + "/drift_study.json",
     "Calibration-drift study."),
    ("qratio_sweep",         PROTO + "/qratio_sweep.json",
     "Measurement-to-data error ratio sweep."),
    ("binary_vs_full_frame", PROTO + "/sixframe_typea.json",
     "Binary {X,Z}^n search vs the full 6^n frame space."),
    ("bias_rate_sweep",      PROTO + "/xyz_psweep_ext.json",
     "Gain across bias strength eta and physical rate p."),
    ("shot_timing",          PROTO + "/bb_shot_timing.json",
     "Per-shot decoder cost, used for the compile-time comparison."),
]


def write_workbook(tables):
    """One sheet per table, plus a README sheet describing each."""
    try:
        import pandas as pd
    except ImportError:
        print("\n  pandas not installed - CSV written, workbook skipped")
        return None
    path = rpath(OUT_DIR, "chameleon_results.xlsx")
    engine = None
    for cand in ("xlsxwriter", "openpyxl"):
        try:
            __import__(cand); engine = cand; break
        except ImportError:
            continue
    if engine is None:
        print("\n  no xlsx engine (pip install xlsxwriter) - CSV written, workbook skipped")
        return None

    doc = dict(DOCS)
    with pd.ExcelWriter(path, engine=engine) as xl:
        # xlsxwriter stamps a creation time into docProps/core.xml, which would make
        # the workbook differ byte-for-byte between otherwise identical runs. Pin it
        # so the whole export is reproducible, not just the CSV.
        if hasattr(xl.book, "set_properties"):        # xlsxwriter; openpyxl has no equivalent
            import datetime as _dt
            xl.book.set_properties({"created": _dt.datetime(2000, 1, 1),
                                    "title": "Chameleon measured results",
                                    "comments": "Generated by data_generator/export_results.py"})
        pd.DataFrame(
            [[n, len(rows), doc.get(n, "")] for n, _, rows in tables],
            columns=["sheet", "rows", "description"],
        ).to_excel(xl, sheet_name="README", index=False)
        for name, header, rows in tables:
            pd.DataFrame(rows, columns=header).to_excel(
                xl, sheet_name=name[:31], index=False)
    print("\n  workbook -> %s  (%d sheets, engine=%s)"
          % (os.path.relpath(path, rpath("")), len(tables) + 1, engine))
    return path


DOCS = {
    "protocol_cells": "Per-cell LER for every policy plus recomputed gains (the main results table).",
    "gain_summary": "Per-code min / mean / median / max of the headline gain.",
    "policy_comparison": "Long-form head-to-head LER, one row per cell x policy.",
    "paper_values": "Every number quoted in the paper, name -> value (685 rows).",
    "table_main_results": "The paper's main results table: per-code anchor LER by policy and gain mean/min/max.",
    "enum_convergence": "Weight-cutoff convergence: operators retained, LUT bytes, enumeration time, regret.",
}
DOCS.update({n: d for n, _, d in SIMPLE})


# Without these the export is not the artifact: they are the measured results
# themselves, not a supporting study. paper_values is produced by a separate
# stage, so its absence is reported as an ordering hint rather than a failure.
REQUIRED = ("protocol_cells", "gain_summary", "policy_comparison", "table_main_results")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-only", action="store_true", help="skip the xlsx workbook")
    ap.add_argument("--allow-partial", action="store_true",
                    help="exit 0 even if a primary result table could not be produced")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    print(">> exporting measured results")
    tables = [t for t in (export_protocol_cells(),
                          export_gain_summary(),
                          export_policy_comparison(),
                          export_main_table(),
                          read_paper_values(),
                          export_enum_convergence()) if t]
    print(">> exporting supporting studies")
    for name, relpath, _ in SIMPLE:
        t = export_simple(name, relpath, _)
        if t:
            tables.append(t)

    # A primary table that came out EMPTY is as broken as one that is absent:
    # load_protocol returns [] when the records cannot be found, so an empty
    # protocol_cells means the reader found nothing, not that nothing was measured.
    nonempty = {name for name, _, rows in tables if rows}
    missing = [t for t in REQUIRED if t not in nonempty]

    if not args.csv_only:
        write_workbook(tables)
    print("\nEXPORT: %d tables -> %s" % (len(tables), os.path.relpath(DATA_DIR, rpath(""))))

    if missing and not args.allow_partial:
        sys.exit(
            "\nincomplete: %d primary result table(s) are missing or empty: %s\n"
            "  The records they read were not found under %s.\n"
            "  Check that LEVER_ROOT points at the artifact root (currently %s)\n"
            "  and that results/ is present. Use --allow-partial to accept a partial export."
            % (len(missing), ", ".join(missing), rpath("results"), rpath("")))


if __name__ == "__main__":
    main()
