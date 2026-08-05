"""Protocol record loading and analysis-time gain recomputation.

This is the data layer every table, macro and export reads. Gains are always
RECOMPUTED here from the stored per-policy masks, never trusted from the
serialized field, so an estimator change cannot leave a stale number behind.
"""
import os, json
from .config import ProtocolConfig, BASELINES

# analysis-time chunk size for cross-fit halves = the protocol sampling chunk
_XFIT_CHUNK = ProtocolConfig().sampling.per
from ._root import rpath

# Where generated tables, macros and exports are written.
OUT_DIR = os.environ.get("CHAM_PAPER_DIR", "output")
PAPER_DIR = OUT_DIR          # backwards-compatible alias


def read_records(path):
    """Load one record file, naming it if it cannot be read.

    A truncated or hand-edited record otherwise surfaces as a bare
    `JSONDecodeError: Expecting property name ... line 1 column 3` with no
    filename -- and the analysis reads dozens of these, so the message says
    nothing about which one to look at.
    """
    try:
        with open(path) as fh:
            recs = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "%s is not valid JSON (%s at line %d column %d). Record files are "
            "written atomically, so a damaged one usually means the file was "
            "edited or a copy was interrupted; restore it with "
            "`git checkout -- %s`."
            % (path, exc.msg, exc.lineno, exc.colno, path)) from exc
    if not isinstance(recs, list):
        raise ValueError("%s holds a %s; a record file must be a JSON list of cells"
                         % (path, type(recs).__name__))
    return recs


def resume_records(path):
    """Records already measured at ``path``, or [] when there are none yet.

    Runners resume by reading their own output, and every one of them wrote
    `try: json.load(open(out)) except Exception: results = []`. That is right for
    a file that does not exist yet and wrong for one that does: a truncated or
    unreadable record was silently treated as EMPTY, so the runner re-measured
    everything from scratch and then overwrote the damaged file -- discarding
    whatever had survived in it, after hours of compute, with nothing printed.
    Only a genuinely absent file starts a fresh run here; anything else stops
    with the message from `read_records`.
    """
    if not os.path.exists(path):
        return []
    return read_records(path)


def load_protocol(cfg=None):
    """All protocol result records: base groups loaded first, deduped by cell key
    (last wins), then cap-extension override files applied globally. Gains are
    RECOMPUTED at analysis time from the stored masks. v1.9.4 censoring rule: an
    UNRESOLVED baseline sits below the Monte-Carlo floor, i.e. below every resolved
    rate, so it would be the true best -- a cell with any unresolved baseline (and
    resolved Cham) reports gain None rather than an inflated min-over-resolved.
    The cross-fitted gain uses the corrected axis-and-baseline cross-fit."""
    cfg = cfg or ProtocolConfig.default()
    key = lambda r: (r["spec"], r["noise"], r["p"], r["fseed"], r["tier"])
    base, exts = {}, []
    import glob as _glob
    for g in "ABC":
        f = rpath(cfg.out_dir, "protocol_v1_%s.json" % g)
        if os.path.exists(f):
            for r in read_records(f):
                base[key(r)] = r
        for fx in sorted(_glob.glob(rpath(cfg.out_dir, "protocol_v1_%s_*ext.json" % g))):
            exts += read_records(fx)
    for r in exts:
        base[key(r)] = r
    out = list(base.values())
    for r in out:
        _recompute_gains(r)
    return out


def _recompute_gains(rec, chunk=None):
    from .estimators import xfit_gain
    m = rec.get("masks")
    if not m or "Cham" not in m:
        return
    names = [k for k in BASELINES if k in m]
    if any(m[k].get("unresolved") for k in names):
        cand = {}          # v1.9.4: unresolved baseline = below-floor true best -> censor
    else:
        cand = {k: m[k]["ler"] for k in names}
    ch = m["Cham"]
    if cand and not ch.get("unresolved") and ch["ler"] > 0:
        bbk = min(cand, key=cand.get)  # exact ties break by fixed tuple order (CSS,XZZX,ZXXZ,Tiurev)
        rec["gain_bb"] = cand[bbk] / ch["ler"]
        rec["gain_css"] = (m["CSS"]["ler"] / ch["ler"]
                           if not m["CSS"].get("unresolved") else None)
    else:
        rec["gain_bb"] = None
        rec["gain_css"] = None
    # xfit_gain returns None for every legitimate censoring case (empty fold half,
    # missing axis, unresolved baseline), so an exception here is a bug and must
    # not be laundered into a None gain.
    rec["gain_xfit"] = xfit_gain(m, _XFIT_CHUNK if chunk is None else chunk)


def headline_gain(rec):
    """The paper's headline estimator with plug-in fallback (v1.9.1 rule)."""
    g = rec.get("gain_xfit")
    return g if g is not None else rec.get("gain_bb")


def resolved(recs):
    """A cell is resolved iff its recomputed gain exists: Cham resolved and at
    least one baseline resolved; under v1.9.4 any unresolved baseline censors the
    cell (shared rule with make_values/make_table)."""
    return [r for r in recs if r.get("gain_bb") is not None]
