"""Cross-entropy searches over the frame space.

Two stages, matching the deployed pipeline:

  cem_pool   binary {X, Z}^n search; returns the top-scoring binary frames, which
             warm-start the full search
  cem6       categorical S3 search over the full 6^n space; returns the best frame
  cem6_pool  the same search, returning the top-K frames instead of just the best

`wt(p)` is the per-qubit cost -log(gamma(p)) shared with the surrogate.
"""
import numpy as np


def wt(p):
    """Per-qubit Bhattacharyya cost -log(2 sqrt(p(1-p))); higher = rarer error.

    NOTE: p here must already be the Y-inclusive marginal (pX+pY or pZ+pY);
    unlike build_U2/build_U6 this does not add pY itself.
    """
    return -np.log(2 * np.sqrt(np.clip(p * (1 - p), 0, None)) + 1e-300)


def cem_pool(Ub, n, iters=15, M=200, seed=5):
    """Binary CEM; returns [(U, D)] ascending (distinct)."""
    rng = np.random.default_rng(seed); x = np.full(n, 0.5); best = []
    for t in range(iters):
        S = (rng.random((M, n)) < x); u = Ub(S)
        idx = np.argsort(u)[:max(2, M // 10)]
        best += [(float(u[i]), S[i].copy()) for i in idx[:4]]
        x = 0.3 * x + 0.7 * S[idx].mean(0); x = np.clip(x, 0.02, 0.98)
    seen = {}
    for u, D in best:
        seen[D.tobytes()] = (u, D)
    return sorted(seen.values(), key=lambda z: z[0])


def cem6(U, n, iters=50, M=500, seed=5, warm=None, restarts=None, elite_frac=0.1):
    """Categorical S3 CEM; returns the top-1 frame (perm indices).
    restarts: list of start distributions (None = uniform); default preserves the
    deployed schedule, two cold + one warm-started. elite_frac: top fraction kept."""
    best = (np.inf, None)
    for ri, st in enumerate(restarts if restarts is not None else [None, None, warm]):
        rng = np.random.default_rng(seed + 101 * ri)
        prob = np.full((n, 6), 1 / 6.0)
        if st is not None:
            prob[:] = 0.3 / 5; prob[np.arange(n), np.asarray(st, int)] = 0.7
        for t in range(iters):
            S = np.array([rng.choice(6, size=M, p=prob[q]) for q in range(n)]).T
            u = U(S); idx = np.argsort(u)[:max(2, int(M * elite_frac))]
            if u[idx[0]] < best[0]:
                best = (float(u[idx[0]]), S[idx[0]].copy())
            emp = np.zeros((n, 6))
            for i in idx:
                emp[np.arange(n), S[i]] += 1
            emp /= len(idx); prob = 0.3 * prob + 0.7 * emp
            prob = np.clip(prob, 0.01, None); prob /= prob.sum(1, keepdims=True)
    if warm is not None:
        uw = float(U(np.asarray(warm, int)[None])[0])
        if uw < best[0]:
            best = (uw, np.asarray(warm, int))
    return best[1]


def cem6_pool(U, n, iters=50, M=500, seed=5, warm=None, topk=96, restarts=None,
              elite_frac=0.1):
    """Same CEM trajectory as cem6, but returns the top-k distinct frames seen."""
    seen = {}
    for ri, st in enumerate(restarts if restarts is not None else [None, None, warm]):
        rng = np.random.default_rng(seed + 101 * ri)
        prob = np.full((n, 6), 1 / 6.0)
        if st is not None:
            prob[:] = 0.3 / 5; prob[np.arange(n), np.asarray(st, int)] = 0.7
        for t in range(iters):
            S = np.array([rng.choice(6, size=M, p=prob[q]) for q in range(n)]).T
            u = U(S)
            for row, uv in zip(S, u):
                seen.setdefault(tuple(int(x) for x in row), float(uv))
            idx = np.argsort(u)[:max(2, int(M * elite_frac))]
            emp = np.zeros((n, 6))
            for i in idx:
                emp[np.arange(n), S[i]] += 1
            emp /= len(idx); prob = 0.3 * prob + 0.7 * emp
            prob = np.clip(prob, 0.01, None); prob /= prob.sum(1, keepdims=True)
    if warm is not None:
        w = np.asarray(warm, int)
        seen.setdefault(tuple(int(x) for x in w), float(U(w[None])[0]))
    return [list(f) for f, _ in sorted(seen.items(), key=lambda kv: kv[1])[:topk]]
