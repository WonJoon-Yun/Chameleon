"""Object model of the Chameleon pipeline.

    cfg  = ProtocolConfig.default()            # or any variant / from_json
    code = Code("surf2d:5")
    fld  = NoiseField("berlin_star", p=cfg.p_anchor, seed=0, code=code)
    sel  = BlindS3Selector(cfg.selector)
    frame = sel.select(code, fld)
    ev   = PhenoEvaluator(cfg, pool)       # pool: multiprocessing.Pool
    res  = ev.worst_axis(code, fld, frame, tier="anchor")

Every numeric choice comes from the config object; swapping a parameter and rerunning
requires no code edits. The math delegates to the verbatim functional modules
(fields/mechs/surrogate/search/pheno), which are golden-hash-tested against the
pre-refactor stack.
"""
from dataclasses import dataclass
import numpy as np

from . import codes as _codes, fields as _fields, mechs as _mechs
from . import surrogate as _surrogate, search as _search, baselines as _baselines, pheno as _pheno
from .config import ProtocolConfig, SelectorConfig


class Code:
    """A CSS code instance: parity checks, logicals, mechanism set (lazy, cached)."""

    def __init__(self, spec):
        self.spec = spec
        self.C, self.rowsX, self.rowsZ = _codes.get_code(spec)
        self.n = self.C["n"]
        self.n_checks = self.C["Hx"].shape[0] + self.C["Hz"].shape[0]
        self._mechs = None

    @property
    def mechs(self):
        """The code's ambiguity operators as (LX, LZ), enumerated once and cached.

        These are the syndrome-free, logically nontrivial operators -- the
        differences between error patterns a decoder cannot tell apart. They
        depend only on the code, never on the noise, so the set is computed on
        first access and reused for every field and frame evaluated on it.
        """
        if self._mechs is None:
            self._mechs = _mechs.mechs(self.spec, self.C)
        return self._mechs

    @property
    def family(self):
        """``"surf"``, ``"color"`` or ``"BB"`` -- which decoder and circuit builder apply."""
        return "surf" if self.spec.startswith("surf") else \
               "color" if self.spec.startswith("color") else "BB"

    @property
    def rounds(self):
        """Syndrome-extraction rounds T for this code under the protocol model (T = d)."""
        return _pheno.DEE[self.spec]

    def __repr__(self):
        return "Code(%s, n=%d)" % (self.spec, self.n)


class NoiseField:
    """One draw of a per-qubit 3-class noise field for a code (data + ancilla triples,
    jointly mean-rescaled to p)."""

    def __init__(self, noise, p, seed, code):
        self.noise, self.p, self.seed = noise, p, seed
        (self.pX, self.pY, self.pZ), (self.aX, self.aY, self.aZ) = \
            _fields.fields_ext(noise, code.n, code.n_checks, seed, p)

    @property
    def rX0(self):
        """Per-qubit rate of errors that flip the X axis, before any frame is applied.

        Y counts here as well as in `rZ0`: a Y error anticommutes with both
        checks, so it is seen by both decoding graphs. The two rates therefore
        do not sum to the total single-qubit rate.
        """
        return self.pX + self.pY

    @property
    def rZ0(self):
        """Per-qubit rate of errors that flip the Z axis, before any frame is applied."""
        return self.pZ + self.pY

    def __repr__(self):
        return "NoiseField(%s, p=%g, seed=%d)" % (self.noise, self.p, self.seed)


class Frame:
    """A per-qubit S3 Clifford frame (permutation indices 0..5)."""

    def __init__(self, perm):
        self.perm = np.asarray(perm, int)

    @classmethod
    def identity(cls, n):
        """The do-nothing frame on ``n`` qubits -- the CSS baseline."""
        return cls(np.zeros(n, int))

    @classmethod
    def from_binary(cls, mask):
        """binary {I,H} mask -> S3: 0 = identity, 1 = the X<->Z swap (2,1,0)."""
        return cls([1 if b else 0 for b in np.asarray(mask, int)])

    def tolist(self):
        """The frame as a plain list of ints, for JSON records and worker arguments."""
        return self.perm.tolist()

    def __eq__(self, other):
        return isinstance(other, Frame) and list(self.perm) == list(other.perm)

    def __repr__(self):
        return "Frame(%s)" % "".join(map(str, self.perm))


def baseline_frames(code, field):
    """The four fixed baselines as Frames."""
    return {k: Frame.from_binary(v)
            for k, v in _baselines.masks_bin(code.C, field.rX0, field.rZ0).items()}


class BlindS3Selector:
    """The protocol selection rule: gamma-product S3 CEM (binary warm start), blind top-1."""

    def __init__(self, cfg: SelectorConfig = None):
        self.cfg = cfg or SelectorConfig()

    def surrogate(self, code, field):
        """Build the batch scorer U6(frames) -> gamma-product scores for this (code, field).

        The returned callable takes an array of frames and scores them all at
        once; lower is better. It closes over the code's ambiguity operators and
        the field's Pauli rates, so it must be rebuilt when either changes.
        """
        LX, LZ = code.mechs
        return _surrogate.build_U6(LX, LZ, code.n, field.pX, field.pY, field.pZ)

    def select(self, code, field):
        """The deployed selection rule: the frame this protocol would use on this cell.

        Blind -- it never measures a LER, so it cannot see the answer it is being
        scored against. Three stages: a binary {I,H} CEM warm start, an S3 CEM
        over the full 6^n space seeded from it, then two guards that decide what
        actually ships. Frames tied in surrogate score are settled by Y-slot mass
        (`_canonicalize`), and the S3 pick is deployed over the binary one only
        when it clears `deploy_margin`; below that the surrogate cannot resolve
        the two, so the binary pick is kept.
        """
        U6 = self.surrogate(code, field)
        c = self.cfg
        Ub = lambda D: U6(np.atleast_2d(D).astype(int))
        warm = np.array(_search.cem_pool(Ub, code.n, iters=c.warm_iters, M=c.warm_pop,
                                         seed=c.cem_seed)[0][1], int)
        pool = _search.cem6_pool(U6, code.n, iters=c.cem_iters, M=c.cem_pop,
                                 seed=c.cem_seed, warm=warm, topk=32,
                                 restarts=[None] * (getattr(c, "cem_restarts", 3) - 1) + [warm],
                                 elite_frac=getattr(c, "cem_elite_frac", 0.1))
        # v1.9.6 tie-breaker: among frames tied in U (measured maps tie exactly when
        # pX=pY collapses the six frames into marginal classes), prefer the one hiding
        # the LEAST rate in the shared Y slot -- Y-slot mass burdens both axes' decoding
        # graphs at once and its apparent U-value is decoder-lottery (BB18 study);
        # remaining ties break lexicographically for determinism.
        from .fields import PERMS
        us = np.array([float(U6(np.asarray(f, int)[None])[0]) for f in pool])
        u0 = us.min()
        P3 = np.stack([field.pX, field.pY, field.pZ], 1)
        best = None
        for f, u in zip(pool, us):
            if u > u0 * (1 + 1e-9):
                continue
            fa = self._canonicalize(np.asarray(f, int), P3)
            ymass = float(P3[np.arange(code.n), [PERMS[v][1] for v in fa]].sum())
            key = (ymass, tuple(int(x) for x in fa))
            if best is None or key < best[0]:
                best = (key, fa)
        # v1.9.7 margin gate (root-caused 2026-07-19): below deploy_margin relative
        # U-improvement the surrogate cannot rank the S3 pick against the binary
        # pick (45% pair inversion at 0-5% gap, 1% beyond 20%), so keep the binary
        # pick; the S3 pick deploys only when it clears the resolution threshold.
        dm = getattr(c, "deploy_margin", 0.0)
        if dm > 0:
            fbin = Frame.from_binary(warm)
            u_bin = float(U6(np.asarray(fbin.tolist(), int)[None])[0])
            u_s3 = float(U6(np.asarray(best[1], int)[None])[0])
            if u_bin > 0 and (u_bin - u_s3) / u_bin < dm:
                return fbin
        return Frame(best[1])

    @staticmethod
    def _canonicalize(F, P3):
        """v1.9.6 per-qubit tie canonicalization: among the perms with an identical
        presented (rX', rZ') pair at a qubit (exact U ties -- on measured maps the
        pX=pY twirl collapses the six frames into three such classes), choose the
        one hiding the least rate in the shared Y slot, then the lowest index.
        The surrogate score is exactly unchanged; only tie-lottery exposure drops."""
        from .fields import PERMS
        F = F.copy()
        for q in range(len(F)):
            p = P3[q]
            rx = {f: p[PERMS[f][0]] + p[PERMS[f][1]] for f in range(6)}
            rz = {f: p[PERMS[f][2]] + p[PERMS[f][1]] for f in range(6)}
            f0 = int(F[q])
            ties = [f for f in range(6)
                    if abs(rx[f] - rx[f0]) <= 1e-15 + 1e-9 * rx[f0]
                    and abs(rz[f] - rz[f0]) <= 1e-15 + 1e-9 * rz[f0]]
            def _yk(f, p=p):   # bind p at def time (closure is used in-iter, but
                y = p[PERMS[f][1]]           # explicit binding is robust to refactor
                return (round(float(y) / max(p.sum(), 1e-300), 9), f)
            F[q] = min(ties, key=_yk)
        return F

    def pool(self, code, field, topk=96):
        """The search's top-``topk`` candidate frames, best surrogate score first.

        The same search `select` runs, without the tie-break and margin gate --
        for studies that need the shape of the candidate set rather than the one
        frame the protocol deploys.
        """
        U6 = self.surrogate(code, field)
        c = self.cfg
        Ub = lambda D: U6(np.atleast_2d(D).astype(int))
        warm = np.array(_search.cem_pool(Ub, code.n, iters=c.warm_iters, M=c.warm_pop,
                                         seed=c.cem_seed)[0][1], int)
        return [Frame(f) for f in _search.cem6_pool(U6, code.n, iters=c.cem_iters,
                                                    M=c.cem_pop, seed=c.cem_seed,
                                                    warm=warm, topk=topk)]


@dataclass
class LerResult:
    """One frame's measured logical error rate on one cell.

    ``ler`` is the worst of the two memory bases, since a frame is only as good
    as the axis it protects least; ``ev_x``/``ev_z`` keep the split. ``unresolved``
    means the shot cap was reached before the tier's event threshold, so this rate
    is a Monte-Carlo floor, not a measurement. The per-chunk failure counts are
    retained so gains can be re-estimated later without re-running anything.
    """

    ler: float
    events: int
    shots: int
    unresolved: bool
    ev_x: int = 0
    ev_z: int = 0
    chunks_z: list = None      # raw per-chunk failure counts, wave-major
    chunks_x: list = None

    def as_dict(self):
        """The record form written to JSON (short keys, as stored in results/)."""
        return dict(ler=self.ler, ev=self.events, N=self.shots, unresolved=self.unresolved,
                    evX=self.ev_x, evZ=self.ev_z, cz=self.chunks_z, cx=self.chunks_x)


_CHUNK_CACHE = {}


def _chunk_setup(spec, noise, fseed, p, perm, mem, dec_cfg):
    """Build-once cell fixtures (circuit + compiled decoder), cached per worker.
    Everything here is seed-independent, so caching cannot change any number."""
    key = (spec, noise, fseed, p, tuple(perm), mem, tuple(sorted(dec_cfg.items())))
    if key in _CHUNK_CACHE:
        return _CHUNK_CACHE[key]
    from ._deps import required
    with required("chromobius", "pymatching", "ldpc",
                  purpose="decoding the measured circuits"):
        import chromobius
        from pymatching import Matching
        from ldpc import BpOsdDecoder
    code = Code(spec)
    fld = NoiseField(noise, p, fseed, code)
    q = p * dec_cfg.get("q_ratio", 1.0)
    circ = _pheno.build_pheno(spec, fld.pX, fld.pY, fld.pZ, q, np.asarray(perm, int), mem)
    if spec.startswith("color"):
        circ = _pheno.add_colors(spec, circ, mem)
        try:
            dem = circ.detector_error_model()
        except ValueError:      # (pX,0,pZ)-type channels need the disjoint approximation
            dem = circ.detector_error_model(approximate_disjoint_errors=True)
        dec = chromobius.compile_decoder_for_dem(dem)
        fix = ("color", circ, dec, None)
    elif spec.startswith("surf"):
        try:
            dem = circ.detector_error_model(decompose_errors=True)
        except ValueError:
            dem = circ.detector_error_model(decompose_errors=True,
                                            approximate_disjoint_errors=True)
        fix = ("surf", circ, Matching.from_detector_error_model(dem), None)
    else:
        H, O, pr = _pheno.dem_matrices(circ)
        dec = BpOsdDecoder(H, error_channel=list(map(float, np.clip(pr, 1e-9, 0.4999))),
                           max_iter=dec_cfg["bp_iters"], bp_method=dec_cfg["bp_method"],
                           schedule=dec_cfg.get("schedule", "serial"),
                           ms_scaling_factor=dec_cfg.get("ms_scaling_factor", 1.0),
                           osd_method=dec_cfg["osd_method"], osd_order=dec_cfg["osd_order"])
        fix = ("BB", circ, dec, O)
    if len(_CHUNK_CACHE) > 8:      # bound worker memory
        _CHUNK_CACHE.clear()
    _CHUNK_CACHE[key] = fix
    return fix


def _decode_chunk(args):
    """Worker: one chunk of one memory basis (module-level for pickling)."""
    spec, noise, fseed, p, perm, mem, sseed, shots, dec_cfg = args
    fam, circ, dec, O = _chunk_setup(spec, noise, fseed, p, perm, mem, dec_cfg)
    det, ob = circ.compile_detector_sampler(seed=sseed).sample(shots, separate_observables=True)
    if fam == "color":
        packed = np.packbits(det, axis=1, bitorder="little")
        pred = dec.predict_obs_flips_from_dets_bit_packed(packed)
        return int(np.sum(pred[:, 0] != ob[:, 0])), shots
    if fam == "surf":
        pred = dec.decode_batch(det)
        return int((pred != ob).any(axis=1).sum()), shots
    # BB: skip trivial syndromes; dedup only when syndromes actually repeat
    detu = det.astype(np.uint8); obu = ob.astype(np.uint8)
    nz = detu.any(axis=1)
    f = int(obu[~nz].any(axis=1).sum())          # empty syndrome -> identity correction
    idx = np.nonzero(nz)[0]
    if len(idx):
        sub = detu[idx]
        uniq, inv = np.unique(sub, axis=0, return_inverse=True)
        if len(uniq) < 0.7 * len(idx):           # dedup pays only under real repetition
            preds = np.zeros((len(uniq), O.shape[0]), dtype=np.uint8)
            for j in range(len(uniq)):
                preds[j] = (O @ dec.decode(uniq[j])) % 2
            f += int((preds[inv] != obu[idx]).any(axis=1).sum())
        else:
            for i in idx:
                pred = (O @ dec.decode(detu[i])) % 2
                if (pred != obu[i]).any():
                    f += 1
    return f, shots


class PhenoEvaluator:
    """Worst-axis LER of a frame under the protocol phenomenological model (q=p, T=d),
    with the protocol sampling schedule and adaptive stopping."""

    def __init__(self, cfg: ProtocolConfig, pool):
        self.cfg = cfg
        self.pool = pool

    def worst_axis(self, code, field, frame, tier="anchor"):
        """Measure a frame on one cell and return its worst-basis LER.

        Samples Z- and X-basis memory in parallel waves and stops as soon as the
        worse basis reaches the tier's event threshold, so easy cells cost little
        and hard ones run to the cap. Reporting the worse basis is what keeps a
        frame from trading one axis away to flatter the other.
        """
        t = self.cfg.tiers[tier]
        s = self.cfg.sampling
        dcfg = dict(bp_iters=self.cfg.decoder.bp_iters, bp_method=self.cfg.decoder.bp_method,
                    schedule=getattr(self.cfg.decoder, "schedule", "serial"),
                    ms_scaling_factor=getattr(self.cfg.decoder, "ms_scaling_factor", 1.0),
                    osd_method=self.cfg.decoder.osd_method, osd_order=self.cfg.decoder.osd_order,
                    q_ratio=getattr(self.cfg, "q_ratio", 1.0))
        fX = fZ = N = 0; wave = 0
        cz, cx = [], []
        # cap semantics: whole waves only -- a cap that is not a multiple of
        # per*nch is overshot by up to one wave (the deployed caps are NOT
        # multiples of per*nch=1.6M, so N runs to the next whole wave above cap;
        # ler=m/N uses the actual N, so this is a shot-count overshoot, not a bias)
        while N < t.cap:
            args = [(code.spec, field.noise, field.seed, field.p, frame.tolist(), mem,
                     s.chunk_seed(wave, k, mem), s.per, dcfg)
                    for mem in ("z", "x") for k in range(s.nch)]
            res = self.pool.map(_decode_chunk, args)
            cz += [f for f, _ in res[:s.nch]]; cx += [f for f, _ in res[s.nch:]]
            fZ += sum(f for f, _ in res[:s.nch]); fX += sum(f for f, _ in res[s.nch:])
            N += s.per * s.nch; wave += 1
            if max(fX, fZ) >= t.minev:
                break
        m = max(fX, fZ)
        return LerResult(m / N, int(m), N, bool(m < t.minev), int(fX), int(fZ), cz, cx)


class Cell:
    """One protocol cell: (code, field, tier) -> all masks measured, gains computed."""

    def __init__(self, cfg, spec, noise, p, fseed, tier):
        self.cfg, self.spec, self.noise, self.p, self.fseed, self.tier = \
            cfg, spec, noise, p, fseed, tier

    def run(self, pool, selector=None):
        """Measure every policy on this cell and return the finished record.

        Runs the four fixed baselines and the selector's frame through the same
        evaluator at the same tier, then computes gains under the censoring rule.
        The selector sees only the code and the noise field -- never a measured
        LER -- so the comparison stays blind.
        """
        code = Code(self.spec)
        field = NoiseField(self.noise, self.p, self.fseed, code)
        sel = selector or BlindS3Selector(self.cfg.selector)
        frames = {k: v for k, v in baseline_frames(code, field).items()}
        frames["Cham"] = sel.select(code, field)
        ev = PhenoEvaluator(self.cfg, pool)
        rec = dict(spec=self.spec, noise=self.noise, p=self.p, fseed=self.fseed,
                   tier=self.tier, model=self.cfg.model,
                   q_ratio=getattr(self.cfg, "q_ratio", 1.0), masks={})
        for nm, fr in frames.items():
            rec["masks"][nm] = ev.worst_axis(code, field, fr, self.tier).as_dict()
        # One implementation of the censoring rule, shared with analysis time: a cell
        # with ANY unresolved baseline reports no gain, because an unresolved rate
        # sits below the Monte-Carlo floor and would otherwise be the apparent best.
        # Re-deriving it here once let the two copies drift.
        from .records import _recompute_gains
        _recompute_gains(rec, chunk=self.cfg.sampling.per)
        return rec


class Matrix:
    """The full experiment matrix of a config, orderable and shardable by code group."""

    TIERS = ("anchor", "dist", "eta", "psweep")

    def __init__(self, cfg: ProtocolConfig):
        self.cfg = cfg

    def cells(self, codes=None):
        """Every cell of the matrix, optionally restricted to ``codes``.

        ``None`` means the whole configured matrix; an EMPTY list means no
        codes and therefore no cells. `codes or cfg.codes` conflated the two, so
        a group whose codes are all retired -- group C is BB144, which the
        deployed matrix no longer contains -- selected the entire 3,752-cell
        matrix instead of nothing, and `make experiments` would have measured
        every code three times over.
        """
        cfg = self.cfg
        codes = list(cfg.codes if codes is None else codes)
        jobs = []
        for ci, spec in enumerate(codes):
            for f in cfg.fields:
                for s in cfg.draws_anchor:
                    jobs.append((0, ci, Cell(cfg, spec, f, cfg.p_anchor, s, "anchor")))
        for ci, spec in enumerate(codes):
            for f in cfg.fields:
                for s in cfg.draws_dist:
                    if s in cfg.draws_anchor:
                        continue
                    jobs.append((1, ci, Cell(cfg, spec, f, cfg.p_anchor, s, "dist")))
        for ci, spec in enumerate(codes):
            for eta in cfg.eta_grid:
                for s in cfg.draws_sweep:
                    jobs.append((2, ci, Cell(cfg, spec, "xyz:%d" % eta, cfg.p_anchor, s, "eta")))
        for ci, spec in enumerate(codes):
            for f in cfg.fields:
                for p in cfg.p_grid:
                    if abs(p - cfg.p_anchor) < 1e-12:
                        continue
                    for s in cfg.draws_sweep:
                        jobs.append((3, ci, Cell(cfg, spec, f, p, s, "psweep")))
        jobs.sort(key=lambda j: (j[0], j[1]))
        return [c for _, _, c in jobs]
