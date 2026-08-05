"""Gamma-product surrogate scores (verbatim from final_protocol.make_U [binary] and
pheno_full_suite.build_U6 [S3], batch-vectorized)."""
import numpy as np


def gamma(r):
    """Bhattacharyya coefficient of a rate-``r`` binary channel: 2*sqrt(r(1-r)).

    The one factor that makes the surrogate decomposable. An ambiguity operator's
    contribution is the product of gamma over its support, so a frame's score is a
    sum of products over per-qubit terms -- computable without ever enumerating
    the error patterns themselves, which is what makes the exact quantity
    intractable. gamma is symmetric about r=1/2 and rises with r, so lowering a
    qubit's presented rate lowers every operator through it.
    """
    return 2 * np.sqrt(np.clip(r * (1 - r), 0, None))


def wt(r):
    """Per-qubit edge weight -log(gamma(r)): products of gamma become sums of wt.

    Lets an operator's contribution be accumulated additively (and searched over)
    in a numerically stable range, rather than multiplying many small factors.
    """
    # physical marginals live in [0, 1/2]; clip so an out-of-range input cannot
    # score as a perfect qubit (gamma would silently hit 0 for r>1)
    return -np.log(gamma(np.clip(r, 0.0, 0.5)) + 1e-300)


def _inc(Ls, n):
    A = np.zeros((len(Ls), n))
    for i, c in enumerate(Ls):
        A[i, list(c)] = 1
    return A



def build_U6(LX, LZ, n, pX, pY, pZ):
    """Full-S3 frame score: rows of F are per-qubit permutation indices 0..5."""
    if not len(LX) or not len(LZ):
        raise ValueError("build_U6: empty mechanism list (LX=%d, LZ=%d)" % (len(LX), len(LZ)))
    from .fields import PERMS
    P3 = np.stack([pX, pY, pZ], 1)
    WX = np.stack([wt(P3[:, PERMS[f][0]] + P3[:, PERMS[f][1]]) for f in range(6)], 1)  # (n,6)
    WZ = np.stack([wt(P3[:, PERMS[f][2]] + P3[:, PERMS[f][1]]) for f in range(6)], 1)
    IX, IZ = _inc(LX, n), _inc(LZ, n)

    def U(F):
        F = np.atleast_2d(np.asarray(F, int))
        gx = np.take_along_axis(np.broadcast_to(WX, (len(F),) + WX.shape), F[:, :, None], 2)[:, :, 0]
        gz = np.take_along_axis(np.broadcast_to(WZ, (len(F),) + WZ.shape), F[:, :, None], 2)[:, :, 0]
        return np.maximum(np.exp(-(gx @ IX.T)).sum(1), np.exp(-(gz @ IZ.T)).sum(1))
    return U
