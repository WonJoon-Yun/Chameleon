"""Neutral (cross-fitted) estimators computed from the stored raw per-chunk counts.

The sampling protocol is unchanged; these functions re-analyze the recorded counts.

Bias background (statistics review, 2026-07-10):
- max(fX,fZ)/N over-estimates the worst-axis rate by ~+6-12% when the two axes are
  close (the max picks the noisier count).
- min over the four baselines under-estimates the best-baseline rate (same selection
  effect, opposite sign), making gains conservative by ~3-9%.

Cross-fitting removes both: the SELECTION (which axis is worse / which baseline is
best) is made on one half of the chunks, the RATE is estimated on the other half,
and the two fold assignments are averaged. Selection and estimation noise are then
independent, so the selection step contributes no FIRST-ORDER bias to the estimate;
when the compared rates are exactly equal any selection is correct, and when they are
far apart the selection is almost surely correct, so that residual (a slight downward
pull at intermediate separations) is bounded and second-order.

Caveat (not removed by cross-fitting): the wave count N is an ADAPTIVE stopping time --
we run whole waves until max(fX,fZ) >= minev on the POOLED counts of both folds -- so
the sample size is correlated with the estimation fold, leaving a sequential-stopping
bias of order 1/minev (~1% at the anchor floor of 100 events, ~3% at the dist floor of
30). It is an upward pull of the same sign for every mask, so it largely cancels in the
reported gain RATIOS and does not flip a gain sign; magnitudes carry it. To eliminate it
rather than bound it, use a fixed shot budget or a negative-binomial (r-1)/(N-1) rate.
"""
import numpy as np
from .config import BASELINES


def _halves(counts):
    c = np.asarray(counts, float)
    return c[0::2].sum(), c[1::2].sum(), len(c[0::2]), len(c[1::2])


def xfit_worst_axis(cz, cx, per):
    """Cross-fitted worst-axis rate from per-chunk counts (chunk size = per shots)."""
    za, zb, nza, nzb = _halves(cz)
    xa, xb, nxa, nxb = _halves(cx)
    if min(nza, nzb, nxa, nxb) == 0:
        return float("nan")   # a fold half is empty (chunk count <= 1) -> 0/0; undefined (cf. xfit_gain)
    # fold 1: select on half A, estimate on half B
    est1 = (xb / (nxb * per)) if (xa / nxa) >= (za / nza) else (zb / (nzb * per))
    # fold 2: select on half B, estimate on half A
    est2 = (xa / (nxa * per)) if (xb / nxb) >= (zb / nzb) else (za / (nza * per))
    return 0.5 * (est1 + est2)


def xfit_gain(masks, per):
    """Cross-fitted gain: BOTH the worst-axis choice and the best-baseline choice
    are made on the selection half; the selected axis rate is estimated on the
    other half; folds averaged. Returns None when the Chameleon mask or the
    selected baseline is unresolved or lacks raw counts (a non-selected baseline
    that lacks counts is dropped from the candidate set)."""
    def halves(c):
        c = np.asarray(c, float)
        return c[0::2], c[1::2]

    def axis_halves(rec):
        if rec.get("cz") is None or rec.get("cx") is None:
            return None
        z0, z1 = halves(rec["cz"]); x0, x1 = halves(rec["cx"])
        if min(len(z0), len(z1), len(x0), len(x1)) == 0:
            return None    # a fold half is empty (chunk count <= 1) -> 0/0 nan; censor
        return ((z0.sum() / (len(z0) * per), x0.sum() / (len(x0) * per)),
                (z1.sum() / (len(z1) * per), x1.sum() / (len(x1) * per)))

    def rate(rec, sel, est):
        h = axis_halves(rec)
        if h is None:
            return None
        ax = 0 if h[sel][0] >= h[sel][1] else 1   # worst axis chosen on sel half
        return h[est][ax]

    if masks["Cham"].get("unresolved"):
        return None
    gains = []
    for sel, est in ((0, 1), (1, 0)):
        cand = {}
        for k in BASELINES:
            if masks[k].get("unresolved"):
                return None    # v1.9.4: below-floor baseline would be the true best
            r = rate(masks[k], sel, sel)
            if r is not None:
                cand[k] = r
        if not cand:
            return None
        bbk = min(cand, key=cand.get)  # exact ties break by fixed tuple order (CSS,XZZX,ZXXZ,Tiurev)
        bb = rate(masks[bbk], sel, est)
        ch = rate(masks["Cham"], sel, est)
        if bb is None or ch is None or bb <= 0 or ch <= 0:
            return None  # censor a zero-event baseline too (symmetric with Cham):
                         # bb=0 would append a spurious gain of 0 and poison min/mean
        gains.append(bb / ch)
    return float(np.mean(gains))
