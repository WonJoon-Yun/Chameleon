"""Fast regression tests for the chameleon package (run: pytest tests/ -q).
The golden circuit hash pins the refactored stack to the pre-refactor stack
(verified bit-identical on 2026-07-10)."""
import sys, os, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import json
import pytest
import numpy as np

GOLDEN_CSS_D3_BERLIN_S0 = "0ee463a46d7b11e3ba6f43b7c5353cce"


def test_golden_circuit_hash():
    import chameleon as ch
    C, _, _ = ch.codes.get_code("surf2d:3")
    n, m = C["n"], C["Hx"].shape[0] + C["Hz"].shape[0]
    (pX, pY, pZ), _ = ch.fields.fields_ext("berlin_star", n, m, 0, 0.005)
    circ = ch.pheno.build_pheno("surf2d:3", pX, pY, pZ, 0.005, np.zeros(n, int), "z")
    assert hashlib.md5(str(circ).encode()).hexdigest() == GOLDEN_CSS_D3_BERLIN_S0


def test_gate_noise_off_by_default_and_frame_invariant():
    """build_pheno's g (gate-noise) knob: default 0 appends nothing (golden circuit unchanged); g>0
    appends one symmetric DEPOLARIZE1 per round, identical across frames -- the frame cannot re-price it."""
    import chameleon as ch
    C, _, _ = ch.codes.get_code("surf2d:3")
    n, m = C["n"], C["Hx"].shape[0] + C["Hz"].shape[0]
    (pX, pY, pZ), _ = ch.fields.fields_ext("berlin_star", n, m, 0, 0.005)
    c0 = ch.pheno.build_pheno("surf2d:3", pX, pY, pZ, 0.005, np.zeros(n, int), "z")
    assert "DEPOLARIZE1" not in str(c0)                      # off by default -> golden hash intact
    g = 0.003
    idF = ch.pheno.build_pheno("surf2d:3", pX, pY, pZ, 0.005, np.zeros(n, int), "z", g=g)
    swF = ch.pheno.build_pheno("surf2d:3", pX, pY, pZ, 0.005, np.ones(n, int), "z", g=g)
    depI = [ln for ln in str(idF).splitlines() if "DEPOLARIZE1" in ln]
    depS = [ln for ln in str(swF).splitlines() if "DEPOLARIZE1" in ln]
    assert len(depI) == 3 and depI == depS                  # one per round (T=d=3); frame-invariant


def test_surrogate_prices_the_circuit_noise():
    """R715 invariant: the surrogate (build_U6) must price EXACTLY the noise the circuit (build_pheno via
    conj1) applies -- both permute (pX,pY,pZ) by PERMS[f]. If they ever diverged, Chameleon would optimize a
    frame the decoder does not measure (silently wrong). Tests build_U6's actual output against conj1's
    presented rates on a single-qubit single-mechanism cell where U(F)=max_axis gamma(presented marginal).
    This pins the PAULI_CHANNEL_1 branch (exact for all-nonzero rates). build_pheno's zero-component branch
    (when a calibration T2>=2*T1 clamps an axis to 0) realizes the channel as independent per-axis errors
    that differ from the priced marginal by a DOCUMENTED O(p^2) cross-term (pheno.py) -- selection-only and
    gain-invariant, so it does not corrupt a reported number."""
    from chameleon.pheno import conj1
    from chameleon.surrogate import build_U6, gamma
    p = np.array([[0.021, 0.005, 0.013]])            # one qubit (pX,pY,pZ), all distinct
    U6 = build_U6([(0,)], [(0,)], 1, p[:, 0], p[:, 1], p[:, 2])
    for f in range(6):
        pxp, pyp, pzp = conj1(list(p[0]), f)         # circuit's presented per-axis rates under frame f
        expected = max(gamma(pxp + pyp), gamma(pzp + pyp))   # 1 mech/1 qubit: U = max-axis gamma(marginal)
        assert abs(float(U6(np.array([[f]]))[0]) - expected) < 1e-9, "surrogate/circuit frame conjugation diverged at f=%d" % f
    assert conj1(list(p[0]), 1)[0] == p[0][2] and conj1(list(p[0]), 1)[2] == p[0][0]   # f=1 = X<->Z swap


def test_selection_deterministic():
    from chameleon.core import Code, NoiseField, BlindS3Selector
    code = Code("surf2d:3")
    fld = NoiseField("xyz:10", 0.005, 0, code)
    sel = BlindS3Selector()
    f1, f2 = sel.select(code, fld), sel.select(code, fld)
    assert f1 == f2


def test_config_roundtrip():
    from chameleon.config import ProtocolConfig
    cfg = ProtocolConfig.default()
    assert ProtocolConfig.from_json(cfg.to_json()) == cfg


def test_frame_binary_embedding():
    from chameleon.core import Frame
    f = Frame.from_binary([0, 1, 1, 0])
    assert f.tolist() == [0, 1, 1, 0]


def test_logops_strict_raises_on_truncation():
    """R250 hardening: on the deployed paths (strict=True) a mechanism-set
    truncation must RAISE, never silently drop mechanisms (the R178 bug class).
    Non-strict keeps the legacy warn-and-return behaviour."""
    import warnings
    from chameleon import mechs
    from chameleon.codes import get_code
    C, _, _ = get_code("surf2d:5")
    n = C["n"]
    # a tiny cap forces truncation; strict=True must raise
    import pytest
    with pytest.raises(RuntimeError):
        mechs.logops(C["Hz"], C["LzZ"], n, extra=2, cap=5, strict=True)
    # non-strict: warns and returns a (truncated) list rather than raising
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = mechs.logops(C["Hz"], C["LzZ"], n, extra=2, cap=5, strict=False)
    assert isinstance(out, list) and len(out) <= 5


def test_add_colors_honors_T_for_code_capacity():
    """R265 regression: add_colors must build detector coords for the ACTUAL round
    count of the circuit, not the hardcoded phenomenological DEE[spec]. The
    code-capacity build (T=1) has 2*n_Z detectors; a DEE=5 coord list mismatches
    and the internal check fires -- which crashed every color cell of Fig-4."""
    import chameleon as ch
    C, _, _ = ch.codes.get_code("color2d:5")
    n = C["n"]
    (pX, pY, pZ), _ = ch.fields.fields_ext("berlin_star", n,
                                           C["Hx"].shape[0] + C["Hz"].shape[0], 0, 0.005)
    circ = ch.pheno.build_pheno("color2d:5", pX, pY, pZ, 0.0,
                                np.zeros(n, int), "z", T=1)   # code capacity
    # T=1 must not raise; the phenomenological default (T=DEE=5) must raise on this circuit
    col = ch.pheno.add_colors("color2d:5", circ, "z", T=1)
    assert col.num_detectors == circ.num_detectors
    import pytest
    with pytest.raises(RuntimeError):   # raise (not assert): survives python -O
        ch.pheno.add_colors("color2d:5", circ, "z")   # wrong T -> coord/detector mismatch


def test_mechanism_depth_converged_at_wmin_plus_2():
    """R275/R276: the code-capacity surrogate enumerates logical-operator mechanisms
    to w_min+2. This locks the ablation finding that deeper mechanisms (w_min+4) do
    NOT change the surrogate ranking -- higher-weight logicals are gamma-suppressed
    (each extra qubit-pair ~ (2 sqrt(p(1-p)))^2 ~ 4p). Uses d=3 for speed. If this
    fails, the w_min+2 cutoff is no longer sufficient and the depth must be revisited."""
    import numpy as np
    import chameleon as ch
    from chameleon import mechs, surrogate
    from chameleon.core import Code, NoiseField
    spec = "surf2d:3"
    C, _, _ = ch.codes.get_code(spec); n = C["n"]
    fld = NoiseField("xyz:10", 0.005, 0, Code(spec))   # strong bias => mechanisms most active

    def U_at(extra):
        LZ = [c for w, c in mechs.logops(C["Hz"], C["LzZ"], n, extra=extra)]
        LX = [c for w, c in mechs.logops(C["Hx"], C["LxX"], n, extra=extra)]
        return surrogate.build_U6(LZ, LX, n, fld.pX, fld.pY, fld.pZ)

    U2, U4 = U_at(2), U_at(4)
    rng = np.random.default_rng(1)
    frames = [np.zeros(n, int)] + [rng.integers(0, 6, n) for _ in range(20)]
    u2 = np.array([float(U2(f[None])[0]) for f in frames])
    u4 = np.array([float(U4(f[None])[0]) for f in frames])
    # argmin unchanged and the deep tail is a tiny fraction of U
    assert int(np.argmin(u2)) == int(np.argmin(u4)), "deeper mechanisms changed the argmin"
    tail = np.max((u4 - u2) / u4)
    assert tail < 0.02, "w_min+4 tail %.3f exceeds 2%% -- w_min+2 may be insufficient" % tail


def test_matrix_counts():
    from chameleon.config import ProtocolConfig
    from chameleon.core import Matrix
    cfg = ProtocolConfig.default()
    cells = Matrix(cfg).cells(["surf2d:3"])
    tiers = {}
    for c in cells:
        tiers[c.tier] = tiers.get(c.tier, 0) + 1
    assert tiers == {"anchor": 20, "dist": 380, "eta": 21, "psweep": 48}


def test_chunk_cache_respects_config():
    """B4: two decoder configs in one process must not share fixtures (q_ratio differs)."""
    from chameleon.core import _decode_chunk, _CHUNK_CACHE
    _CHUNK_CACHE.clear()
    base = dict(bp_iters=1000, bp_method="minimum_sum", osd_method="OSD_CS",
                osd_order=7, q_ratio=1.0)
    a = ("surf2d:3", "berlin_star", 0, 0.005, [0] * 9, "z", 1100, 20000, base)
    b = ("surf2d:3", "berlin_star", 0, 0.005, [0] * 9, "z", 1100, 20000,
         dict(base, q_ratio=0.0))
    fa, _ = _decode_chunk(a)
    fb, _ = _decode_chunk(b)
    assert len(_CHUNK_CACHE) == 2          # distinct fixtures per config
    assert fa != fb                         # q=0 removes measurement noise -> fewer failures


# ---------------------------------------------------------------------------
# Decoder-config plumbing
# ---------------------------------------------------------------------------

def test_decoder_cfg_defaults_are_serial_product_sum():
    from chameleon.config import ProtocolConfig
    dec = ProtocolConfig().decoder
    assert dec.schedule == "serial"
    assert dec.bp_method == "product_sum"
    assert dec.osd_method == "OSD_CS"
    assert dec.osd_order == 7
    assert dec.bp_iters == 1000


def test_worst_axis_assembles_dec_cfg(monkeypatch):
    """worst_axis must forward the full decoder config (incl. schedule and q_ratio)
    to every worker-chunk args tuple."""
    import chameleon.core as core
    from chameleon.config import ProtocolConfig, TierConfig

    captured = []

    def fake_decode_chunk(args):
        captured.append(args[-1])          # dcfg dict is the last element
        return 0, args[7]                  # (fails, shots)

    monkeypatch.setattr(core, "_decode_chunk", fake_decode_chunk)

    class SerialPool:
        def map(self, f, xs):
            return list(map(f, xs))

    cfg = ProtocolConfig()
    # minev=0: stopping condition max(fX,fZ) >= minev holds after the first wave
    # even with zero failures, so exactly one wave runs.
    cfg.tiers["anchor"] = TierConfig(minev=0, cap=1)
    code = core.Code("surf2d:3")
    field = core.NoiseField("xyz:10", 0.005, 0, code)
    frame = core.Frame.identity(9)
    res = core.PhenoEvaluator(cfg, SerialPool()).worst_axis(code, field, frame, tier="anchor")

    # one wave = nch chunks per axis, two axes
    assert len(captured) == 2 * cfg.sampling.nch
    assert res.shots == cfg.sampling.per * cfg.sampling.nch
    for dcfg in captured:
        assert dcfg["schedule"] == "serial"
        assert dcfg["bp_method"] == "product_sum"
        assert dcfg["q_ratio"] == 1.0


# ---------------------------------------------------------------------------
# Drift noise field
# ---------------------------------------------------------------------------

def test_drift_field_parses_and_is_deterministic():
    from chameleon import fields

    a = fields.field3x("drift:0.3:7:xyz:10", 50, seed=0, P=0.005)
    b = fields.field3x("drift:0.3:7:xyz:10", 50, seed=0, P=0.005)
    for xa, xb in zip(a, b):
        assert np.array_equal(xa, xb)      # deterministic

    base = fields.field3x("xyz:10", 50, seed=0, P=0.005)
    assert any(not np.array_equal(xa, xb) for xa, xb in zip(a, base))

    # sigma=0: rng.normal(0, 0.0) is exactly 0.0 -> factor exp(0)=1.0 -> bit-identical
    zero = fields.field3x("drift:0:7:xyz:10", 50, seed=0, P=0.005)
    for xz, xb in zip(zero, base):
        assert np.array_equal(xz, xb)

    other = fields.field3x("drift:0.3:8:xyz:10", 50, seed=0, P=0.005)
    assert any(not np.array_equal(xa, xo) for xa, xo in zip(a, other))


# ---------------------------------------------------------------------------
# Fixed-total-rate bias model. The historical failure mode (see the note
# corrections) was a field whose per-qubit total drifts off P, which biases the
# robust qubits and fabricates gain. These pin total==P and the eta ratio.
# ---------------------------------------------------------------------------

def test_synthetic_fields_have_fixed_total_rate_p():
    from chameleon import fields
    P = 0.005
    for noise in ("xyz:10", "half:10", "xyz:100", "half:5", "xyz:1"):
        pX, pY, pZ = fields.field3x(noise, 300, seed=1, P=P)
        assert np.allclose(pX + pY + pZ, P, atol=1e-12), noise


def test_xyz_field_eta_ratio_and_one_dominant_axis():
    from chameleon import fields
    P, eta = 0.005, 10.0
    pX, pY, pZ = fields.field3x("xyz:10", 300, seed=2, P=P)
    stk = np.sort(np.stack([pX, pY, pZ]), axis=0)   # [lo, lo, hi] per qubit
    assert np.allclose(stk[0], stk[1])              # exactly two equal low axes
    assert np.allclose(stk[2] / stk[0], eta)        # dominant/low == eta
    assert np.allclose(stk[2], P * eta / (eta + 2))


def test_half_field_has_zero_Y_and_eta_imbalance():
    from chameleon import fields
    P, eta = 0.005, 10.0
    pX, pY, pZ = fields.field3x("half:10", 300, seed=3, P=P)
    assert np.allclose(pY, 0.0)
    hi = np.maximum(pX, pZ); lo = np.minimum(pX, pZ)
    assert np.allclose(hi / lo, eta)


def test_xyz_eta1_is_depolarizing():
    """The eta=1 anchor must be exactly depolarizing (every frame ties at 1x)."""
    from chameleon import fields
    pX, pY, pZ = fields.field3x("xyz:1", 200, seed=4, P=0.005)
    assert np.allclose(pX, pY) and np.allclose(pY, pZ)


def test_noisefield_device_global_mean_total_is_p():
    """Device-derived fields resample per-qubit and rescale so the mean total over
    ALL qubits (data + ancilla) is exactly P. The data-only and ancilla-only means
    legitimately differ -- the rescale is global, so e.g. Berlin data runs hot and
    Willow data runs cool while the operating point stays P."""
    import numpy as np
    from chameleon.core import Code, NoiseField
    code = Code("surf2d:5")
    for dev in ("berlin_star", "miami_star", "willow_star"):
        fld = NoiseField(dev, 0.005, 0, code)
        allq = np.concatenate([fld.pX + fld.pY + fld.pZ, fld.aX + fld.aY + fld.aZ])
        assert abs(allq.mean() - 0.005) < 1e-9, dev


# ---------------------------------------------------------------------------
# Protocol loading: cap-extension overrides
# ---------------------------------------------------------------------------

def _proto_rec(spec, noise, p, fseed, tier, base_ler, cham_ler, per=20000, nch=8):
    """Minimal valid protocol record: masks carry ler + raw per-chunk counts whose
    rates reproduce ler exactly (so gain_bb and gain_xfit agree)."""
    def mask(ler):
        c = [int(round(ler * per))] * nch
        return dict(ler=ler, unresolved=False, cz=list(c), cx=list(c))
    masks = {k: mask(base_ler) for k in ("CSS", "XZZX", "ZXXZ", "Tiurev")}
    masks["Cham"] = mask(cham_ler)
    return dict(spec=spec, noise=noise, p=p, fseed=fseed, tier=tier,
                masks=masks, gain_bb=999.0)   # bogus stored gain: must be recomputed


def test_override_replaces_base_cells_once(tmp_path):
    import json
    from chameleon.config import ProtocolConfig
    from chameleon.records import load_protocol

    base = [_proto_rec("surf2d:3", "xyz:10", 0.005, 0, "anchor", 0.010, 0.005),
            _proto_rec("surf2d:3", "xyz:10", 0.005, 1, "anchor", 0.010, 0.005)]
    ext = [_proto_rec("surf2d:3", "xyz:10", 0.005, 0, "anchor", 0.020, 0.002),  # replaces
           _proto_rec("surf2d:7", "xyz:10", 0.005, 0, "anchor", 0.010, 0.001)]  # new
    json.dump(base, open(tmp_path / "protocol_v1_A.json", "w"))
    json.dump(ext, open(tmp_path / "protocol_v1_A_d7ext.json", "w"))

    cfg = ProtocolConfig()
    cfg.out_dir = str(tmp_path)            # absolute path: rpath() passes it through
    recs = load_protocol(cfg)

    assert len(recs) == 3                  # 2 base, 1 replaced in place, 1 new
    key = lambda r: (r["spec"], r["noise"], r["p"], r["fseed"], r["tier"])
    by = {key(r): r for r in recs}
    hit = by[("surf2d:3", "xyz:10", 0.005, 0, "anchor")]
    assert hit["masks"]["Cham"]["ler"] == 0.002          # override masks won
    assert abs(hit["gain_bb"] - 0.020 / 0.002) < 1e-12   # recomputed, not the stored 999
    assert abs(by[("surf2d:3", "xyz:10", 0.005, 1, "anchor")]["gain_bb"] - 2.0) < 1e-12
    assert abs(by[("surf2d:7", "xyz:10", 0.005, 0, "anchor")]["gain_bb"] - 10.0) < 1e-12


# ---------------------------------------------------------------------------
# Cross-fitted gain estimator
# ---------------------------------------------------------------------------

def _xfit_masks(base_cz, base_cx, cham_cz, cham_cx, base_unres=False, cham_unres=False):
    def mask(cz, cx, unres):
        return dict(ler=0.0, unresolved=unres, cz=list(cz), cx=list(cx))
    m = {k: mask(base_cz, base_cx, base_unres) for k in ("CSS", "XZZX", "ZXXZ", "Tiurev")}
    m["Cham"] = mask(cham_cz, cham_cx, cham_unres)
    return m


def test_xfit_gain_known_answer():
    from chameleon.estimators import xfit_gain
    per = 20000
    # symmetric counts: both folds give 100/10 = 10
    m = _xfit_masks([100] * 8, [100] * 8, [10] * 8, [10] * 8)
    g = xfit_gain(m, per)
    assert abs(g - 10.0) < 1e-9
    # Cham X axis worse (20 > 10): worst axis X is chosen on the selection half and
    # its rate is estimated on the other half; both folds identical -> 100/20 = 5
    m = _xfit_masks([100] * 8, [100] * 8, [10] * 8, [20] * 8)
    g = xfit_gain(m, per)
    assert abs(g - 5.0) < 1e-9


def test_xfit_worst_axis_direct():
    """The standalone worst-axis estimator selects and estimates on opposite halves."""
    from chameleon.estimators import xfit_worst_axis
    # X axis is worse (0.02 vs 0.01); both folds agree, so the cross-fit returns 0.02
    assert abs(xfit_worst_axis([10, 10], [20, 20], 1000) - 0.02) < 1e-12


def test_xfit_gain_best_baseline_crossfit():
    """The best-of-four-baselines choice must be made on the selection half and
    estimated on the OTHER half, so a baseline that looks lucky-low on the selection
    half cannot also supply its own (biased) rate estimate. Known answer 1.75; a bug
    that selected and estimated on the same half would return 1.0."""
    from chameleon.estimators import xfit_gain

    def mask(cz, cx):
        return dict(ler=0.0, unresolved=False, cz=cz, cx=cx)
    # CSS is lowest on the even (selection) chunks but high on the odd (estimate) chunks
    masks = {
        "CSS": mask([5, 20, 5, 20], [5, 20, 5, 20]),
        "XZZX": mask([15, 15, 15, 15], [15, 15, 15, 15]),
        "ZXXZ": mask([15, 15, 15, 15], [15, 15, 15, 15]),
        "Tiurev": mask([15, 15, 15, 15], [15, 15, 15, 15]),
        "Cham": mask([10, 10, 10, 10], [10, 10, 10, 10]),
    }
    assert abs(xfit_gain(masks, 1000) - 1.75) < 1e-9


def test_gain_none_when_one_baseline_unresolved():
    """Any single unresolved baseline censors the cell (a below-floor baseline could
    be the true best), not only the all-unresolved case."""
    from chameleon.estimators import xfit_gain

    def mask(cz, cx, unres=False):
        return dict(ler=0.0, unresolved=unres, cz=cz, cx=cx)
    masks = {k: mask([10] * 8, [10] * 8) for k in ("CSS", "XZZX", "ZXXZ", "Tiurev")}
    masks["ZXXZ"]["unresolved"] = True          # exactly one baseline unresolved
    masks["Cham"] = mask([5] * 8, [5] * 8)
    assert xfit_gain(masks, 1000) is None


def test_gain_none_when_fold_empty():
    """A chunk list of length <= 1 leaves one cross-fit fold half empty (0/0 nan);
    the estimator must censor the cell to None rather than emit a NaN gain."""
    from chameleon.estimators import xfit_gain

    def mask(cz, cx):
        return dict(ler=0.0, unresolved=False, cz=cz, cx=cx)
    masks = {k: mask([10], [10]) for k in ("CSS", "XZZX", "ZXXZ", "Tiurev")}
    masks["Cham"] = mask([5], [5])
    assert xfit_gain(masks, 1000) is None


def test_gain_none_when_unresolved():
    from chameleon.estimators import xfit_gain
    per = 20000
    # Cham unresolved
    m = _xfit_masks([100] * 8, [100] * 8, [10] * 8, [10] * 8, cham_unres=True)
    assert xfit_gain(m, per) is None
    # every baseline unresolved
    m = _xfit_masks([100] * 8, [100] * 8, [10] * 8, [10] * 8, base_unres=True)
    assert xfit_gain(m, per) is None
    # Cham counts all zero -> ch <= 0
    m = _xfit_masks([100] * 8, [100] * 8, [0] * 8, [0] * 8)
    assert xfit_gain(m, per) is None


# ---------------------------------------------------------------------------
# Sampling seed schedule
# ---------------------------------------------------------------------------

def test_chunk_seed_schedule_is_injective():
    from chameleon.config import SamplingConfig
    s = SamplingConfig()
    seeds = [s.chunk_seed(w, c, mem)
             for w in range(25) for c in range(25) for mem in ("z", "x")]
    assert len(seeds) == len(set(seeds)) == 25 * 25 * 2
    for w in range(25):
        for c in range(25):
            assert s.chunk_seed(w, c, "x") - s.chunk_seed(w, c, "z") == 500


# ---------------------------------------------------------------------------
# v1.9.6 tie-canonicalization (_canonicalize): the pY-tiebreaker.
# It may only permute a qubit's frame within an exact presented-(rX,rZ) tie
# class, so it must (a) never change any presented rate, (b) pick the least-Y
# member of the tie class, (c) be idempotent, (d) collapse a depolarizing qubit
# to the identity frame.
# ---------------------------------------------------------------------------

def _presented(F, P3):
    from chameleon.fields import PERMS
    rx = np.array([P3[q][PERMS[F[q]][0]] + P3[q][PERMS[F[q]][1]] for q in range(len(F))])
    rz = np.array([P3[q][PERMS[F[q]][2]] + P3[q][PERMS[F[q]][1]] for q in range(len(F))])
    return rx, rz


def _yslot(f, p):
    from chameleon.fields import PERMS
    return p[PERMS[f][1]]


def test_canonicalize_preserves_presented_rates():
    """The surrogate sees only (rX,rZ); canonicalization must leave them exact."""
    from chameleon.core import BlindS3Selector
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = 8
        P3 = rng.random((n, 3)) * 0.02
        F = rng.integers(0, 6, n)
        Fc = BlindS3Selector._canonicalize(F, P3)
        rx0, rz0 = _presented(F, P3)
        rx1, rz1 = _presented(Fc, P3)
        assert np.allclose(rx0, rx1) and np.allclose(rz0, rz1)


def test_canonicalize_minimizes_y_slot_in_tie_class():
    """Per qubit, the canonical frame hides the least rate in the shared Y slot
    among all frames with the same presented (rX,rZ) -- the pY-tiebreaker."""
    from chameleon.core import BlindS3Selector
    from chameleon.fields import PERMS
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = 6
        # include twirled (pX=pY and pX=pZ) qubits where real ties exist
        P3 = rng.random((n, 3)) * 0.02
        P3[0] = [P3[0][0], P3[0][0], P3[0][2]]   # pX=pY
        P3[1] = [P3[1][0], P3[1][1], P3[1][0]]   # pX=pZ
        F = rng.integers(0, 6, n)
        Fc = BlindS3Selector._canonicalize(F, P3)
        for q in range(n):
            p = P3[q]
            rx = {f: p[PERMS[f][0]] + p[PERMS[f][1]] for f in range(6)}
            rz = {f: p[PERMS[f][2]] + p[PERMS[f][1]] for f in range(6)}
            f0 = int(F[q])
            ties = [f for f in range(6)
                    if abs(rx[f] - rx[f0]) <= 1e-15 + 1e-9 * rx[f0]
                    and abs(rz[f] - rz[f0]) <= 1e-15 + 1e-9 * rz[f0]]
            assert _yslot(int(Fc[q]), p) <= min(_yslot(f, p) for f in ties) + 1e-15


def test_canonicalize_idempotent():
    from chameleon.core import BlindS3Selector
    rng = np.random.default_rng(2)
    P3 = rng.random((10, 3)) * 0.02
    F = rng.integers(0, 6, 10)
    Fc = BlindS3Selector._canonicalize(F, P3)
    assert np.array_equal(Fc, BlindS3Selector._canonicalize(Fc, P3))


def test_canonicalize_depolarizing_is_identity():
    """A depolarizing qubit (all axes equal) has all six frames tied; the least-Y,
    lowest-index choice is the identity frame 0."""
    from chameleon.core import BlindS3Selector
    P3 = np.full((5, 3), 0.007)
    F = np.array([3, 1, 4, 5, 2])
    Fc = BlindS3Selector._canonicalize(F, P3)
    assert np.array_equal(Fc, np.zeros(5, int))


def test_e_column_ecell_branches():
    """Pin the load-bearing E-column logic (e_column.ecell): the same/TBD/gain/missing
    decision that sets each Chameleon-E cell of the hero table (see R804: a stale/wrong
    value here = a wrong Table-IV cell; the freshness guard catches staleness, not a
    logic regression). ecell lives in scripts/, so add it to the path."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    from e_column import ecell

    # (a) no matching seeds -> missing
    assert ecell({}, "surf2d:5", "willow_star") == ("", "missing")

    # (b) all infeasible -> TBD  (e.g. BB72 anchor: search can't resolve the low LER)
    grid_infeas = {"BB72|berlin_star|%d" % s: {"feasible": False} for s in range(3)}
    assert ecell(grid_infeas, "BB72", "berlin_star") == ("TBD", "infeasible")

    # (c) >=1 feasible, none switched -> same  (the BB36-Berlin case fixed in R804)
    grid_same = {
        "BB36|berlin_star|0": {"feasible": True, "deployed_evo": False, "evo_gain": 0.78},
        "BB36|berlin_star|1": {"feasible": False},
    }
    assert ecell(grid_same, "BB36", "berlin_star") == ("same", "same")

    # (d) a switch -> gain, formatted mean [min,max] percent
    grid_gain = {
        "BB18|willow_star|0": {"feasible": True, "deployed_evo": True, "evo_gain": 1.10},
        "BB18|willow_star|1": {"feasible": True, "deployed_evo": True, "evo_gain": 1.20},
    }
    txt, kind = ecell(grid_gain, "BB18", "willow_star")
    assert kind == "gain"
    assert txt == "+15\\%\\,[+10,+20]", txt

    # cross-field isolation: the BB36 grid must not leak into a BB18 query
    assert ecell(grid_same, "BB18", "berlin_star") == ("", "missing")


def test_xfit_gain_censors_zero_event_baseline():
    """bb<=0 branch (estimators.py:90, previously untested per R859 agent review): the
    selected (min) baseline has zero events on the estimation half -> xfit_gain returns
    None (censor), NOT a spurious gain of 0 that would poison the min/mean."""
    from chameleon.estimators import xfit_gain
    # CSS is the min baseline on the even (sel=0) half but has 0 events on the odd half;
    # on the (sel=0, est=1) fold bb is estimated on the all-zero odd half -> bb=0 -> None.
    masks = {
        "Cham":   {"cz": [1, 1, 1, 1], "cx": [1, 1, 1, 1]},
        "CSS":    {"cz": [1, 0, 1, 0], "cx": [1, 0, 1, 0]},
        "XZZX":   {"cz": [5, 5, 5, 5], "cx": [5, 5, 5, 5]},
        "ZXXZ":   {"cz": [5, 5, 5, 5], "cx": [5, 5, 5, 5]},
        "Tiurev": {"cz": [5, 5, 5, 5], "cx": [5, 5, 5, 5]},
    }
    assert xfit_gain(masks, 100) is None


# ---------------------------------------------------------------- artifact docs
# The README is the artifact's contract with an evaluator: it names the tables the
# export produces and their row counts. Those numbers drifted once already (the
# gain_summary regrouping), so they are pinned here rather than trusted.

@pytest.fixture(scope="session")
def export_dir(tmp_path_factory):
    """A complete export, built for the test session into a throwaway directory.

    These tests used to read whatever `output/` happened to hold and skip when it
    was absent -- so on a fresh clone the entire export contract went unchecked
    while pytest still reported success. Building it here costs about seven
    seconds from the shipped records (no simulator, no decoder) and buys three
    things: the tests always run, they read a freshly built export rather than a
    stale one, and because the destination is NOT the default they also prove
    CHAM_PAPER_DIR is honoured end to end.
    """
    import subprocess, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dest = tmp_path_factory.mktemp("export")
    env = dict(os.environ, CHAM_PAPER_DIR=str(dest), LEVER_ROOT=root,
               PYTHONPATH=os.path.join(root, "src"))
    for stage in ("export_results.py", "paper_values.py"):
        r = subprocess.run([sys.executable, os.path.join(root, "data_generator", stage)]
                           + (["--csv-only"] if stage == "export_results.py" else []),
                           cwd=root, env=env, capture_output=True, text=True)
        assert r.returncode == 0, "%s failed:\n%s\n%s" % (stage, r.stdout[-2000:], r.stderr[-2000:])
    data = os.path.join(str(dest), "data")
    assert os.path.isdir(data), "the export produced no data directory in CHAM_PAPER_DIR"
    return data


def test_export_honours_a_redirected_output_dir(export_dir):
    """Every table the export writes lands in CHAM_PAPER_DIR -- all of them.

    paper_values.py hardcoded `output/data/paper_values.csv` while
    export_results.py resolved its destination through records.OUT_DIR, so a
    redirected run split its output: 16 tables where they were asked for and the
    17th silently back inside the artifact, overwriting whatever was there. The
    README advertises CHAM_PAPER_DIR as the way to keep output outside the
    artifact, so this was a documented promise the code half-kept.
    """
    import glob
    produced = sorted(os.path.basename(f) for f in glob.glob(os.path.join(export_dir, "*.csv")))
    assert len(produced) == 17, "expected 17 tables in CHAM_PAPER_DIR, got %d: %s" % (
        len(produced), produced)
    assert "paper_values.csv" in produced, "paper_values.csv did not follow CHAM_PAPER_DIR"


def test_export_fails_loudly_on_a_wrong_root(tmp_path):
    """A mis-set LEVER_ROOT must fail with an actionable message, not a quiet
    export of empty tables.

    load_protocol returns [] when it cannot find the records, so every primary
    table would otherwise be written with a header and no rows, and the run would
    report success -- the failure mode this guards.
    """
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, LEVER_ROOT=str(tmp_path),
               PYTHONPATH=os.path.join(root, "src"))
    r = subprocess.run([sys.executable,
                        os.path.join(root, "data_generator", "export_results.py"),
                        "--csv-only"], capture_output=True, text=True, env=env)
    assert r.returncode != 0, "an empty root exported successfully:\n" + r.stdout
    msg = r.stdout + r.stderr
    assert "missing or empty" in msg, msg[-400:]
    assert "LEVER_ROOT" in msg, "the message does not say what to check:\n" + msg[-400:]

    ok = subprocess.run([sys.executable,
                         os.path.join(root, "data_generator", "export_results.py"),
                         "--csv-only", "--allow-partial"],
                        capture_output=True, text=True, env=env)
    assert ok.returncode == 0, "--allow-partial did not opt out:\n" + ok.stderr


def test_censoring_rule_has_one_implementation():
    """Cell.run must not re-derive the gain rule that records.py owns.

    The two copies agreed on every shipped cell only because an unresolved mask
    carries the Monte-Carlo floor, which is the minimum rate, so min() selected it
    and the guard fired. That is a property of the floor convention, not of the
    rule, so the duplicate was a latent divergence.
    """
    import inspect
    from chameleon import core
    src = inspect.getsource(core.Cell.run)
    assert "_recompute_gains" in src, "Cell.run no longer delegates the gain rule"
    assert 'rec["gain_bb"] =' not in src, "Cell.run computes gain_bb itself again"
    assert 'rec["gain_css"] =' not in src, "Cell.run computes gain_css itself again"


def test_unresolved_baseline_censors_the_gain():
    """Any unresolved baseline censors the cell, even when another is the minimum."""
    from chameleon.records import _recompute_gains

    def mask(ler, unresolved=False):
        return dict(ler=ler, ev=100, N=10 ** 6, unresolved=unresolved,
                    evX=50, evZ=50, cz=[10] * 8, cx=[10] * 8)

    # XZZX is unresolved but NOT the minimum, so a min-first rule would miss it
    rec = {"masks": {"CSS": mask(2e-3), "XZZX": mask(9e-3, unresolved=True),
                     "ZXXZ": mask(1.5e-3), "Tiurev": mask(1.8e-3), "Cham": mask(1e-3)}}
    _recompute_gains(rec)
    assert rec["gain_bb"] is None, "an unresolved non-minimal baseline was not censored"
    assert rec["gain_css"] is None

    rec = {"masks": {"CSS": mask(2e-3), "XZZX": mask(1.6e-3), "ZXXZ": mask(1.5e-3),
                     "Tiurev": mask(1.8e-3), "Cham": mask(1e-3)}}
    _recompute_gains(rec)
    assert rec["gain_bb"] == pytest.approx(1.5)      # min baseline 1.5e-3 / 1e-3
    assert rec["gain_css"] == pytest.approx(2.0)


def test_code_spec_errors_name_the_valid_options():
    """A typo in a code spec must say what is valid, not raise a bare KeyError."""
    from chameleon.codes import get_code
    for bad in ("surf2D:5", "surface:5", "BB99", "surf2d:x", "", 5):
        with pytest.raises(ValueError) as e:
            get_code(bad)
        msg = str(e.value)
        assert repr(bad) in msg or str(bad) in msg, msg
        assert any(hint in msg for hint in ("surf2d", "BB72", "non-empty")), msg
    for bad, why in (("surf2d:4", "odd"), ("surf2d:0", "odd")):
        with pytest.raises(ValueError, match=why):
            get_code(bad)


def test_out_of_range_rate_is_rejected_everywhere():
    """p outside (0, 1) must raise, on every noise family.

    Unvalidated it does not fail: gamma clips a negative r(1-r) to zero, so an
    impossible rate reads as a perfect qubit and the surrogate returns a finite,
    optimistic score. Measured before the guard: p=5.0 gave per-qubit totals up
    to 8.81 and U(CSS)=1.3e+01 with no warning.
    """
    from chameleon.fields import field3x
    families = ("willow_star", "berlin_star", "xyz:10", "half:5",
                "xyzmix:10:0.5", "drift:0.1:0:willow_star")
    for noise in families:
        for bad in (-1.0, 0.0, 1.0, 5.0):
            with pytest.raises(ValueError, match="out of range"):
                field3x(noise, 9, 0, bad)
        pX, pY, pZ = field3x(noise, 9, 0, 0.005)      # the valid case still works
        assert np.all(pX >= 0) and np.all(pX + pY + pZ <= 1)


def test_unknown_noise_field_lists_the_families():
    from chameleon.fields import field3x
    with pytest.raises(ValueError) as e:
        field3x("berlin", 9, 0, 0.005)                # missing the _star suffix
    assert "_star" in str(e.value) and "willow" in str(e.value), str(e.value)


# ---------------------------------------------------------------- estimators
def test_headline_gain_prefers_crossfit_and_falls_back():
    """The paper's headline number is the cross-fitted gain, with the plug-in as
    fallback. Every reported gain goes through this, so the precedence is pinned."""
    from chameleon.records import headline_gain
    assert headline_gain({"gain_xfit": 1.5, "gain_bb": 2.0}) == 1.5
    assert headline_gain({"gain_xfit": None, "gain_bb": 2.0}) == 2.0
    assert headline_gain({"gain_bb": 2.0}) == 2.0
    assert headline_gain({"gain_xfit": None, "gain_bb": None}) is None
    assert headline_gain({}) is None


def test_resolved_keeps_exactly_the_cells_with_a_gain():
    """resolved() is the censoring filter behind gain_summary; it must key on
    gain_bb, which _recompute_gains sets to None for a censored cell."""
    from chameleon.records import resolved
    recs = [{"gain_bb": 1.2}, {"gain_bb": None}, {"gain_bb": 0.8}, {}]
    assert resolved(recs) == [{"gain_bb": 1.2}, {"gain_bb": 0.8}]


# ---------------------------------------------------------------- baselines
def test_fixed_baselines_ignore_the_noise_map_and_tiurev_does_not():
    """The paper's claim about prior work, pinned as a property of the code.

    XZZX and ZXXZ are fixed coordinate checkerboards: the same frame regardless of
    the calibration. The local rule reads the map, so it differs between maps. CSS
    is the undeformed frame. If a refactor ever made XZZX noise-dependent, every
    comparison in the paper would change meaning without any test failing.
    """
    from chameleon.baselines import masks_bin
    from chameleon.codes import get_code
    from chameleon.fields import field3x

    C, _, _ = get_code("surf2d:5")
    n = C["n"]
    frames = []
    for seed in (0, 1):
        pX, pY, pZ = field3x("willow_star", n, seed, 0.005)
        frames.append(masks_bin(C, pX + pY, pZ + pY))

    a, b = frames
    assert not a["CSS"].any(), "CSS is the undeformed frame"
    assert np.array_equal(a["CSS"], b["CSS"])
    assert np.array_equal(a["XZZX"], b["XZZX"]), "XZZX changed with the noise map"
    assert np.array_equal(a["ZXXZ"], b["ZXXZ"]), "ZXXZ changed with the noise map"
    assert np.array_equal(a["XZZX"], 1 - a["ZXXZ"]), "XZZX and ZXXZ are not complementary"
    assert not np.array_equal(a["Tiurev"], b["Tiurev"]), \
        "the local rule did not react to a different noise map"


# ---------------------------------------------------------------- search
def test_cem_searches_are_deterministic_and_improve_on_the_undeformed_frame():
    """The deployed selector must be reproducible from its seed and must not
    return a frame worse than CSS on the objective it is minimising."""
    from chameleon.codes import get_code
    from chameleon.mechs import mechs
    from chameleon.fields import field3x
    from chameleon.surrogate import build_U6
    from chameleon.search import cem6

    spec = "surf2d:3"
    C, _, _ = get_code(spec)
    n = C["n"]
    LX, LZ = mechs(spec, C)
    pX, pY, pZ = field3x("willow_star", n, 3, 0.005)
    U = build_U6(LX, LZ, n, pX, pY, pZ)

    a = cem6(U, n, iters=6, M=80, seed=0)
    b = cem6(U, n, iters=6, M=80, seed=0)
    assert np.array_equal(a, b), "same seed produced a different frame"

    u_css = float(U(np.zeros(n, int))[0])
    u_sel = float(U(np.asarray(a, int)[None])[0])
    assert u_sel <= u_css, "the search returned a frame worse than the undeformed code"


def test_runner_re_measures_instead_of_refusing_on_a_config_mismatch(tmp_path):
    """`make experiments` must start on a fresh checkout.

    The shipped records were produced under an earlier config revision, so their
    cfg_id does not match the current one. The runner used to exit 1 with
    "refusing to resume", which made the documented full re-measurement fail
    immediately. It now writes a fresh, config-tagged file and leaves the shipped
    records untouched; --strict-resume restores the refusal.
    """
    import subprocess, json as _j
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, LEVER_ROOT=str(tmp_path),
               PYTHONPATH=os.path.join(root, "src"), GROUP="A", PROCS="1",
               CHAM_TIERS="")
    # An empty code list selects no cells, so this exercises the resume guard
    # without measuring anything. (Naming a bogus tier used to do the job, but
    # --tier is validated now, which is the point of the sibling test.)
    cfgfile = tmp_path / "empty.json"
    cfgfile.write_text(_j.dumps({"codes": []}))
    out = tmp_path / "results" / "protocol_v1"
    out.mkdir(parents=True)
    (out / "protocol_v1_A.json").write_text(_j.dumps(
        [{"spec": "surf2d:3", "noise": "willow_star", "p": 0.005, "fseed": 0,
          "tier": "anchor", "cfg_id": "v1.9:deadbeef", "masks": {}}]))

    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "run_protocol.py"),
                        "--config", str(cfgfile)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, "the runner refused to start:\n" + r.stdout + r.stderr
    assert "re-measuring from scratch into" in r.stdout, r.stdout
    assert (out / "protocol_v1_A.json").read_text().count("deadbeef") == 1, \
        "the shipped records were modified"

    strict = subprocess.run([sys.executable, os.path.join(root, "scripts", "run_protocol.py"),
                             "--config", str(cfgfile), "--strict-resume"],
                            capture_output=True, text=True, env=env)
    assert strict.returncode != 0, "--strict-resume did not refuse"
    assert "refusing to resume" in strict.stdout + strict.stderr


def test_every_flag_the_shell_drivers_pass_is_accepted():
    """A driver must not pass a flag the script does not have.

    `make quick` passed `--tier anchor` to a runner whose argparse had no --tier,
    so the documented quick re-measurement exited 2 with "unrecognized arguments"
    before doing anything. Parse the drivers and check each flag against the
    script's own --help.
    """
    import re, subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"), LEVER_ROOT=root)

    # flags the drivers pass, including ones held in a shell variable
    wanted = {}
    for driver in ("reproduce_all.sh", "reproduce_results.sh", "smoke.sh"):
        text = open(os.path.join(root, "scripts", driver)).read()
        for script in re.findall(r"python3 ((?:scripts|data_generator)/[a-z_0-9/]+\.py)", text):
            wanted.setdefault(script, set())
        for var, val in re.findall(r'^(\w+)="(--[^"]*)"', text, re.M):
            for script in wanted:
                if "$" + var in text:
                    wanted[script] |= set(re.findall(r"--[a-z-]+", val))
        for script, flags in re.findall(
                r"python3 ((?:scripts|data_generator)/[a-z_0-9/]+\.py)((?: --[a-z-]+)*)", text):
            wanted[script] |= set(re.findall(r"--[a-z-]+", flags))

    missing = []
    for script, flags in wanted.items():
        if not flags:
            continue
        helptext = subprocess.run([sys.executable, os.path.join(root, script), "--help"],
                                  capture_output=True, text=True, env=env).stdout
        for f in sorted(flags):
            if f not in helptext:
                missing.append("%s does not accept %s" % (script, f))
    assert not missing, "; ".join(missing)


def test_tier_selection_is_validated():
    """--tier must reject an unknown tier and name the valid ones."""
    from chameleon.core import Matrix
    assert set(Matrix.TIERS) == {"anchor", "dist", "eta", "psweep"}

    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"), LEVER_ROOT=root,
               GROUP="A", CHAM_TIERS="")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "run_protocol.py"),
                        "--tier", "ancor"], capture_output=True, text=True, env=env)
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "unknown tier" in out and "anchor" in out, out


def test_a_stale_lock_is_taken_over_but_a_live_one_is_not(tmp_path):
    """An interrupted run must not block its output forever.

    atexit does not run when a job is killed, so a killed runner left a .lock
    behind and every later run of that script died with a raw FileExistsError --
    while the README documents the runners as safe to interrupt. Four such locks
    were found in the shipped tree, all owned by dead pids.
    """
    from chameleon._root import acquire_lock
    target = tmp_path / "out.json"

    stale = tmp_path / "out.json.lock"
    stale.write_text("999999999")                 # a pid that cannot be running
    acquire_lock(str(target))                     # must take it over, not raise
    assert stale.exists() and stale.read_text() == str(os.getpid())

    os.remove(stale)
    stale.write_text(str(os.getpid()))            # a live owner: this process
    with pytest.raises(SystemExit) as e:
        acquire_lock(str(target))
    msg = str(e.value)
    assert str(os.getpid()) in msg and "lock" in msg, msg
    os.remove(stale)


def test_the_experiment_driver_reports_failures_instead_of_skipping_them():
    """reproduce_all.sh must not turn a crashed study into "(skipped)" and exit 0."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "scripts", "reproduce_all.sh")).read()
    assert "(skipped:" not in text, "a failing study is still reported as skipped"
    assert "FAILED" in text and "exit 1" in text, \
        "the driver does not exit non-zero when a study fails"


def test_a_study_rerun_does_not_clobber_the_shipped_record(tmp_path):
    """Re-running a study must leave the shipped record in place.

    The records under results/ are the evidence the reported numbers came from.
    Every study wrote its output with atomic_json_dump straight over that path,
    so a reviewer checking reproducibility would destroy the data the paper
    quotes, with no way back except git. (This happened during the audit and was
    caught only by the byte-identity check.)
    """
    from chameleon import _root
    from chameleon._root import atomic_json_dump

    rec = tmp_path / "rec.json"
    rec.write_text('{"shipped": 1}')

    _root._WRITTEN.clear()
    out = atomic_json_dump({"new": 1}, str(rec))
    assert out != str(rec), "the shipped record was overwritten"
    assert json.loads(rec.read_text()) == {"shipped": 1}, "shipped content changed"
    assert out.endswith(".rerun.json")

    again = atomic_json_dump({"new": 2}, str(rec))     # a study that appends
    assert again == out, "a second dump was redirected somewhere else"
    assert json.loads(open(out).read()) == {"new": 2}

    _root._WRITTEN.clear()
    forced = atomic_json_dump({"forced": 1}, str(rec), allow_overwrite=True)
    assert forced == str(rec) and json.loads(rec.read_text()) == {"forced": 1}

    _root._WRITTEN.clear()
    fresh = tmp_path / "brand_new.json"
    assert atomic_json_dump({"a": 1}, str(fresh)) == str(fresh)


def test_docker_runner_is_usable_non_interactively():
    """docker/run.sh must work from a pipe, a CI job or a script.

    Four defects made the documented reviewer path fail, none of which show up
    when the script is only read:
      -it was unconditional        -> "cannot attach stdin to a TTY-enabled
                                      container" whenever stdout is not a tty
      -u was unconditional         -> rootless docker maps container-root to the
                                      invoking user already, and a host uid
                                      outside the subuid range fails with
                                      "setgroups: invalid argument"
      "${@:-make smoke}"           -> the default was passed as ONE argv entry,
                                      so docker looked for an executable named
                                      "make smoke"
      "${TTY[@]}" when TTY=()      -> macOS ships bash 3.2, where expanding an
                                      EMPTY array is an "unbound variable" error
                                      under `set -u` (bash 4.4 made the empty
                                      case safe). Both arrays are empty in the
                                      ordinary case, so the script aborted on
                                      line 46 before reaching docker -- and it
                                      aborted precisely when there was no tty,
                                      i.e. in the non-interactive use this test
                                      is named for. ${arr[@]+"${arr[@]}"} is the
                                      portable form: still an argv array, but it
                                      expands to nothing when empty.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # comments quote the old forms to explain them, so inspect code lines only
    code = "\n".join(l for l in open(os.path.join(root, "docker", "run.sh"))
                     if not l.lstrip().startswith("#"))
    assert "-t 0" in code and "-t 1" in code, "-it is not conditional on a terminal"
    assert "rootless" in code, "the user flag is not conditional on rootless docker"
    assert '${@:-' not in code, "the default command is still passed as a single argument"
    assert "set -- make smoke" in code, "no argv-list default command"
    assert 'exec docker run --rm ${TTY[@]+"${TTY[@]}"} ${USERFLAG[@]+"${USERFLAG[@]}"}' in code, \
        "flags are not applied as argv arrays, or not in the bash-3.2-safe form"


def test_an_empty_code_list_selects_no_cells():
    """`codes=[]` means no codes, not "the whole matrix".

    Matrix.cells used `codes or cfg.codes`, so an empty selection fell through to
    the full configuration. Group C is BB144, which the deployed matrix no longer
    contains, so `GROUP=C run_protocol.py` selected all 3,752 cells instead of
    none: `make experiments` would have measured 6,566 cells rather than 2,814,
    with group C re-measuring every code into the BB144 group's file.
    """
    from chameleon.config import ProtocolConfig
    from chameleon.core import Matrix

    cfg = ProtocolConfig.default()
    m = Matrix(cfg)
    whole = len(m.cells(None))
    assert whole > 0
    assert len(m.cells([])) == 0, "an empty code list still selected cells"
    assert len(m.cells(None)) == whole, "None no longer means the whole matrix"
    one = len(m.cells(["surf2d:3"]))
    assert 0 < one < whole


def test_every_driver_group_selects_only_its_own_codes():
    """No group may select a code outside its own membership."""
    import importlib.util, warnings
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    spec = importlib.util.spec_from_file_location(
        "rp", os.path.join(root, "scripts", "run_protocol.py"))
    mod = importlib.util.module_from_spec(spec)
    warnings.simplefilter("ignore")
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass

    from chameleon.config import ProtocolConfig
    from chameleon.core import Matrix
    cfg = ProtocolConfig.default()
    seen = []
    for group, members in mod.GROUPS.items():
        sel = [c for c in cfg.codes if c in members]
        cells = Matrix(cfg).cells(sel)
        assert {c.spec for c in cells} <= set(members), \
            "group %s selected codes outside %s" % (group, members)
        seen += sel
    assert sorted(seen) == sorted(cfg.codes), \
        "the groups do not partition the deployed matrix: %s vs %s" % (sorted(seen), sorted(cfg.codes))


def test_exported_rows_say_whether_their_code_is_in_the_paper(export_dir):
    """The records hold measurements of codes retired from the matrix.

    BB30, BB108 and color2d:7 account for 490 of the 2,896 exported cells and
    reach the aggregates (24 of 161 gain_summary rows, 9 of 41 in the main
    table), so a reviewer comparing table_main_results.csv against the paper
    would find rows the paper does not report. The rows are kept -- they are real
    measurements -- and carry a `deployed` flag instead.
    """
    import csv as _csv
    from chameleon.config import ProtocolConfig

    data = export_dir

    deployed = set(ProtocolConfig.default().codes)
    for name in ("protocol_cells", "gain_summary", "policy_comparison", "table_main_results"):
        with open(os.path.join(data, "%s.csv" % name)) as fh:
            rows = list(_csv.DictReader(fh))
        assert rows, name
        assert "deployed" in rows[0], "%s has no deployed column" % name
        wrong = [r["code"] for r in rows
                 if (r["deployed"] == "True") != (r["code"] in deployed)]
        assert not wrong, "%s mislabels %s" % (name, sorted(set(wrong))[:4])
        assert any(r["deployed"] == "False" for r in rows), \
            "%s has no retired rows: the flag would be untested" % name


def test_no_shell_script_assigns_a_reserved_variable():
    """A driver must not name a variable the shell already owns.

    reproduce_all.sh used GROUPS, which in bash is a builtin array holding the
    caller's group IDs. The assignment silently did nothing, so `for g in
    $GROUPS` iterated over numeric GIDs and every run of `make experiments` died
    with KeyError: '1007000513' before measuring anything.
    """
    import re, glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    special = {"GROUPS", "SECONDS", "RANDOM", "LINENO", "PWD", "OLDPWD", "UID", "EUID",
               "PPID", "HOSTNAME", "SHELL", "SHLVL", "IFS", "FUNCNAME", "BASH", "BASHPID",
               "REPLY", "OPTARG", "OPTIND", "HISTFILE", "COLUMNS", "LINES"}
    bad = []
    for path in (glob.glob(os.path.join(root, "*.sh"))
                 + glob.glob(os.path.join(root, "scripts", "*.sh"))
                 + glob.glob(os.path.join(root, "docker", "*.sh"))):
        for i, line in enumerate(open(path), 1):
            m = re.match(r"\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)=", line)
            if m and m.group(1) in special:
                bad.append("%s:%d assigns %s" % (os.path.basename(path), i, m.group(1)))
    assert not bad, "; ".join(bad)


def test_unknown_execution_group_is_rejected_with_the_valid_ones():
    """GROUP=Z must not reach a KeyError inside main()."""
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(root, "src"), LEVER_ROOT=root,
               GROUP="Z", CHAM_TIERS="")
    r = subprocess.run([sys.executable, os.path.join(root, "scripts", "run_protocol.py")],
                       capture_output=True, text=True, env=env)
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "unknown execution group" in out and "A, B, C" in out, out
    assert "KeyError" not in out, "still failing with a raw KeyError:\n" + out


def test_shell_scripts_are_clean_under_shellcheck():
    """Every shell script must pass shellcheck.

    The drivers are part of the artifact's interface and their failures only show
    up when run: a bash builtin silently shadowed a variable (pass 14), an
    unconditional -it broke every non-interactive use (pass 11). Static analysis
    catches the rest. Skipped when shellcheck is not installed, so the suite
    still runs on a bare checkout.
    """
    import glob, shutil, subprocess
    # shellcheck-py ships only the binary (no importable module), so PATH is the
    # single place to look -- an `import shellcheck_py` fallback can never fire.
    sc = shutil.which("shellcheck")
    if sc is None:
        pytest.skip("shellcheck not installed (pip install -e '.[dev]')")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scripts = sorted(glob.glob(os.path.join(root, "*.sh"))
                     + glob.glob(os.path.join(root, "scripts", "*.sh"))
                     + glob.glob(os.path.join(root, "docker", "*.sh")))
    assert scripts, "no shell scripts found"
    r = subprocess.run([sc, "-f", "gcc"] + scripts, capture_output=True, text=True)
    assert not r.stdout.strip(), "shellcheck findings:\n" + r.stdout


def test_multi_word_flags_are_passed_as_argv_arrays():
    """A flag pair held in a variable must not become one argv entry.

    "--tier anchor" as a single argument is exactly how `make quick` failed
    before, and "${@:-make smoke}" is how docker/run.sh failed. Quoting a string
    variable reintroduces both; an array is the fix, so the arrays are pinned.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ra = open(os.path.join(root, "scripts", "reproduce_all.sh")).read()
    assert "TIER=()" in ra and '"${TIER[@]}"' in ra, \
        "reproduce_all.sh no longer passes --tier as an array"
    assert 'TIER="--tier anchor"' not in ra

    run = open(os.path.join(root, "docker", "run.sh")).read()
    assert "TTY=()" in run and "USERFLAG=()" in run, "docker/run.sh flags are not arrays"
    # ${arr[@]+"${arr[@]}"} rather than a bare "${arr[@]}": still an argv array,
    # but it survives `set -u` on bash 3.2 when the array is empty. See
    # test_docker_runner_is_usable_non_interactively.
    assert '${TTY[@]+"${TTY[@]}"} ${USERFLAG[@]+"${USERFLAG[@]}"}' in run


def test_no_dead_imports_or_locals():
    """The package and drivers must be pyflakes-clean.

    Dead code is not merely untidy in an artifact: a name imported but never
    used reads as a dependency that does not exist, a variable assigned but
    never read reads as a computation that matters, and a comment describing a
    check the function does not perform misleads whoever is trying to reproduce
    it. All three were present. Skipped when pyflakes is absent so a bare
    checkout still passes.
    """
    import glob, importlib.util, subprocess, sys
    if importlib.util.find_spec("pyflakes") is None:
        pytest.skip("pyflakes not installed (pip install pyflakes)")

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sorted(glob.glob(os.path.join(root, "src", "chameleon", "*.py"))
                   + glob.glob(os.path.join(root, "src", "chameleon", "vendor", "*.py"))
                   + glob.glob(os.path.join(root, "scripts", "*.py"))
                   + glob.glob(os.path.join(root, "data_generator", "*.py"))
                   + glob.glob(os.path.join(root, "tests", "*.py")))
    assert files
    r = subprocess.run([sys.executable, "-m", "pyflakes"] + files,
                       capture_output=True, text=True)
    # pyflakes cannot resolve `from x import *`; that note is not a finding.
    out = "\n".join(l for l in r.stdout.splitlines()
                    if l.strip() and "unable to detect undefined names" not in l)
    assert not out, "pyflakes findings:\n" + out


def test_every_make_target_is_discoverable_and_resolvable():
    """`make help` lists every target, and every target it lists actually resolves.

    The help text used to be `sed -n '2,12p' Makefile` -- pinned by line number,
    not content -- so it silently fell behind the targets it was supposed to
    index: 4 of 13 were undiscoverable, including `quick`, which the README does
    advertise. An evaluator reads `make help` to find out what the artifact can
    do; anything missing from it does not exist as far as they are concerned.
    """
    import re, subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    makefile = open(os.path.join(root, "Makefile")).read()
    phony = re.search(r"^\.PHONY:(.*)$", makefile, re.M).group(1).split()
    assert len(phony) > 5, "no .PHONY target list found"

    helptext = subprocess.run(["make", "help"], cwd=root, capture_output=True,
                              text=True).stdout
    missing = [t for t in phony if not re.search(r"\bmake +%s\b" % re.escape(t), helptext)]
    assert not missing, "targets absent from `make help`: %s" % " ".join(missing)

    listed = set(re.findall(r"^\s*make +([a-zA-Z_-]+)", helptext, re.M))
    assert listed <= set(phony), "`make help` advertises targets that do not exist: %s" % (
        sorted(listed - set(phony)))

    # a target whose recipe cannot even be expanded is a broken promise
    for t in phony:
        r = subprocess.run(["make", "-n", t], cwd=root, capture_output=True, text=True)
        assert r.returncode == 0, "`make %s` does not resolve:\n%s" % (t, r.stderr[-800:])


def test_readme_only_promises_targets_that_exist():
    """Every `make <target>` the README tells an evaluator to run is a real target."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    makefile = open(os.path.join(root, "Makefile")).read()
    phony = set(re.search(r"^\.PHONY:(.*)$", makefile, re.M).group(1).split())
    readme = open(os.path.join(root, "README.md")).read()
    promised = set(re.findall(r"\bmake +([a-z][a-z-]*)\b", readme))
    assert promised, "README names no make targets"
    assert promised <= phony, "README promises non-existent targets: %s" % sorted(promised - phony)


def test_every_script_documents_itself_and_answers_help():
    """Each script has a real module docstring and a CLI that handles --help.

    Nine of them carried their description as a bare string placed AFTER the
    first import, which makes it an expression Python evaluates and discards --
    not a docstring. It was invisible to `--help`, `pydoc` and `help()`, and ten
    scripts had no argument parsing at all, so `--help` fell through to the
    module body and launched a multi-hour measurement. Both together meant the
    first thing anyone types at an unfamiliar script did the worst possible
    thing.
    """
    import ast, glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    missing = []
    for f in sorted(glob.glob(os.path.join(root, "scripts", "*.py"))
                    + glob.glob(os.path.join(root, "data_generator", "*.py"))):
        src = open(f).read()
        rel = os.path.relpath(f, root)
        if ast.get_docstring(ast.parse(src)) is None:
            missing.append(rel + ": no module docstring")
        if "parse_no_args" not in src and "ArgumentParser" not in src:
            missing.append(rel + ": no argument parsing, so --help would run the script")
    assert not missing, "\n".join(missing)


def test_help_is_answered_without_doing_the_work(tmp_path):
    """--help exits 0 and prints the description, on a script that otherwise measures.

    Run for real rather than inspected: the failure mode being pinned is a
    script that parses fine and still ignores argv.
    """
    import subprocess, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, LEVER_ROOT=root, PYTHONPATH=os.path.join(root, "src"),
               CHAM_PAPER_DIR=str(tmp_path))
    for script in ("binary_ablation.py", "abstention_analysis.py"):
        r = subprocess.run([sys.executable, os.path.join(root, "scripts", script), "--help"],
                           cwd=root, env=env, capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, "%s --help exited %d:\n%s" % (script, r.returncode, r.stderr[-800:])
        assert "usage:" in r.stdout, "%s --help printed no usage" % script
        body = r.stdout.split("usage:")[-1]
        assert len(body.strip().splitlines()) > 3, \
            "%s --help printed a bare usage line, no description" % script
        # a typo must be refused, not silently ignored and then measured for hours
        r2 = subprocess.run([sys.executable, os.path.join(root, "scripts", script), "--typo"],
                            cwd=root, env=env, capture_output=True, text=True, timeout=300)
        assert r2.returncode != 0, "%s accepted an unknown argument" % script
        assert "unrecognized" in r2.stderr, "%s gave no reason for rejecting it" % script


def test_readme_contents_block_matches_the_tree():
    """Every path the Contents block draws exists, and the stage numbering is consistent.

    The block called scripts/ "Stage 1" and data_generator/ "Stage 2" while the
    section that DEFINES the stages says the opposite -- an evaluator reading top
    to bottom is told to run the core-days job to get the ten-minute result.
    """
    import glob, re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    readme = open(os.path.join(root, "README.md")).read()
    block = readme.split("## Contents", 1)[1].split("```", 2)[1]

    # output/ is generated by `make all`; absent on a fresh checkout by design
    GENERATED = {"output"}
    SEARCH = ("", "src/chameleon", "scripts", "data_generator")

    missing = []
    for line in block.splitlines():
        m = re.search(r"[\u251c\u2514]\u2500\u2500 ([A-Za-z0-9_.*/,-]+)", line)
        if not m:
            continue
        for name in re.split(r",\s*", m.group(1).strip()):
            name = name.strip().rstrip("/")
            if not name or name in GENERATED:
                continue
            cands = [os.path.join(root, d, name) for d in SEARCH]
            if any(glob.glob(c) for c in cands):
                continue
            missing.append(name)
    assert not missing, "Contents block draws paths that do not exist: %s" % sorted(set(missing))

    # the stage labels in the tree must agree with the stage headings below it
    tree_stage = dict(re.findall(r"([a-z_]+)/\s+Stage (\d)", block))
    assert tree_stage.get("data_generator") == "1", (
        "Contents calls data_generator/ Stage %s; the Reproduction section defines Stage 1 as "
        "the 10-minute regeneration from records" % tree_stage.get("data_generator"))
    assert tree_stage.get("scripts") == "2", (
        "Contents calls scripts/ Stage %s; the Reproduction section defines Stage 2 as the "
        "full re-measurement" % tree_stage.get("scripts"))


def test_a_missing_measurement_dependency_says_how_to_install_it():
    """Importing a simulator-backed module without the stack names the tier and the fix.

    The README tells an evaluator to install the MINIMAL tier -- numpy and
    pandas -- because that is all the reproduction path needs. The likely next
    thing they try is a study script, which landed on a bare
    `ModuleNotFoundError: No module named 'panqec'` and a traceback through code
    they have never read, with nothing to say the package is optional or which
    requirements file supplies it.
    """
    import importlib, sys

    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] == "panqec":
                raise ModuleNotFoundError("No module named 'panqec'", name="panqec")
            return None

    dropped = {k: v for k, v in sys.modules.items()
               if k == "chameleon.codes" or k.startswith("panqec")}
    for k in dropped:
        del sys.modules[k]
    sys.meta_path.insert(0, Block())
    try:
        with pytest.raises(ModuleNotFoundError) as ei:
            importlib.import_module("chameleon.codes")
        msg = str(ei.value)
        assert "measurement stack" in msg, msg
        assert "requirements.txt" in msg, "the message does not name the file that fixes it"
        assert "make all" in msg, "the message does not say the reproduction path is unaffected"
    finally:
        sys.meta_path.pop(0)
        sys.modules.update(dropped)

    # a package outside the measurement tier must not be disguised as a tier problem
    from chameleon._deps import required
    with pytest.raises(ModuleNotFoundError) as ei2:
        with required("panqec"):
            raise ModuleNotFoundError("No module named 'definitely_not_a_dep'",
                                      name="definitely_not_a_dep")
    assert "measurement stack" not in str(ei2.value)


def test_a_damaged_record_file_is_named(tmp_path):
    """A record that will not parse reports WHICH file and how to restore it."""
    import json
    from chameleon.records import read_records

    good = tmp_path / "ok.json"
    good.write_text(json.dumps([{"spec": "surf2d:3"}]))
    assert read_records(str(good)) == [{"spec": "surf2d:3"}]

    bad = tmp_path / "protocol_v1_A.json"
    bad.write_text("{ truncated")
    with pytest.raises(ValueError) as ei:
        read_records(str(bad))
    msg = str(ei.value)
    assert str(bad) in msg, "the error does not name the file: " + msg
    assert "git checkout" in msg, "the error does not say how to recover: " + msg

    notalist = tmp_path / "wrong.json"
    notalist.write_text('{"spec": "surf2d:3"}')
    with pytest.raises(ValueError) as ei3:
        read_records(str(notalist))
    assert "list of cells" in str(ei3.value)


def test_the_reproduction_path_imports_no_simulator():
    """`make all` must run with the whole measurement stack absent.

    This is the README's central promise -- reproduce on numpy and pandas alone --
    and it is what the lazy submodule loading exists to deliver, so it is checked
    by running the export with every simulator import forced to fail.
    """
    import subprocess, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    blocker = (
        "import sys, runpy\n"
        "BLOCK = {'stim','pymatching','ldpc','chromobius','panqec','qldpc'}\n"
        "class B:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BLOCK:\n"
        "            raise ModuleNotFoundError('blocked', name=name.split('.')[0])\n"
        "        return None\n"
        "sys.meta_path.insert(0, B())\n"
        "sys.argv = [sys.argv[1], '--csv-only']\n"
        "runpy.run_path(sys.argv[0], run_name='__main__')\n")
    script = os.path.join(root, "data_generator", "export_results.py")
    r = subprocess.run([sys.executable, "-c", blocker, script], cwd=root,
                       env=dict(os.environ, LEVER_ROOT=root,
                                PYTHONPATH=os.path.join(root, "src"),
                                CHAM_PAPER_DIR=str(tmp_path_for_blocked())),
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, "the export needs a simulator:\n%s\n%s" % (
        r.stdout[-1500:], r.stderr[-1500:])
    assert "EXPORT:" in r.stdout, r.stdout[-800:]


def tmp_path_for_blocked():
    import tempfile
    return tempfile.mkdtemp(prefix="cham-blocked-")


def test_documented_stage_costs_agree_across_the_docs():
    """The same command must not carry two different costs in two places.

    `bash docker/run.sh` was "~10 min" in docker/README.md and "~5 min" in the
    script's own header, for the identical command; the main README said "~5 min"
    for the same smoke run and "~10 min" for `make all`. Measured on the
    reference machine, smoke is ~72 s and `make all` ~12 s -- so the figures an
    evaluator provisions against were wrong by more than an order of magnitude
    AND disagreed with each other. Times are approximate by nature; what this
    pins is that every place quoting a stage quotes the SAME figure.
    """
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = ["README.md", "docker/README.md", "docker/run.sh", "Makefile", "scripts/smoke.sh"]

    def costs(pattern):
        found = {}
        for rel in files:
            text = open(os.path.join(root, rel)).read()
            for line in text.splitlines():
                if re.search(pattern, line) and "~" in line:
                    m = re.search(r"~\s*([\d.]+)\s*(s|sec|seconds|min|minutes)", line)
                    if m:
                        n, unit = float(m.group(1)), m.group(2)
                        found.setdefault(round(n * (60 if unit.startswith("min") else 1)), []).append(rel)
        return found

    for label, pattern in (("smoke", r"kick the tires|Stage 0"),
                           ("make all", r"make all|Stage 1 —")):
        seen = costs(pattern)
        assert len(seen) <= 1, (
            "%s is documented with conflicting costs: %s"
            % (label, {("%gs" % k): v for k, v in seen.items()}))


def test_docker_docs_describe_the_scripts_that_exist():
    """Every flag and compose service docker/README.md tells the reader to use is real."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    doc = open(os.path.join(root, "docker", "README.md")).read()
    build = open(os.path.join(root, "docker", "build.sh")).read()
    compose = open(os.path.join(root, "docker", "docker-compose.yml")).read()

    for flag in sorted(set(re.findall(r"build\.sh[^\n`]*?(--[a-z-]+)", doc))):
        assert re.search(r"%s\)" % re.escape(flag), build), \
            "docker/README.md documents `build.sh %s`, which build.sh does not accept" % flag

    services = set(re.findall(r"^  ([a-z][a-z-]*):", compose, re.M))
    for svc in sorted(set(re.findall(r"docker-compose\.yml run --rm ([a-z-]+)", doc))):
        assert svc in services, \
            "docker/README.md documents compose service %r, which is not defined (have: %s)" % (
                svc, sorted(services))


def _pyproject_deps():
    """(required, extras) dependency name lists, parsed without a TOML library.

    tomllib is 3.11+, and this package supports 3.9 -- a test that silently
    skipped on the declared floor would be no gate at all.
    """
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "pyproject.toml")).read()

    def arr(pattern):
        m = re.search(pattern + r"\s*=\s*\[(.*?)\]", text, re.S | re.M)
        if not m:
            return None
        return sorted(re.split(r"[><=!\[]", s, 1)[0].strip()
                      for s in re.findall(r'"([^"]+)"', m.group(1)))

    required = arr(r"^dependencies")
    extras = {name: arr(r"^" + name) for name in ("measure", "dev")}
    return required, extras


def _requirements(path):
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for line in open(os.path.join(root, path)):
        line = line.split("#")[0].strip()
        if line:
            out.append(re.split(r"[><=!]", line, 1)[0].strip())
    return sorted(out)


def test_installing_the_package_does_not_pull_in_the_simulator_stack():
    """pyproject's required dependencies are the reproduction tier, nothing more.

    It declared the whole measurement stack -- stim, panqec, qldpc, chromobius,
    ldpc, pymatching, scipy, sympy -- as REQUIRED, which contradicts the
    artifact's central design: submodules load lazily so reproducing every
    reported result runs on numpy and pandas alone. `pip install chameleon-qec`
    would have dragged in eight simulator packages for someone who only wanted
    to read the shipped records.
    """
    required, extras = _pyproject_deps()
    assert required is not None, "no [project] dependencies array found"
    assert required == _requirements("requirements-minimal.txt"), (
        "pyproject's required dependencies and requirements-minimal.txt disagree:\n"
        "  pyproject: %s\n  minimal:   %s" % (required, _requirements("requirements-minimal.txt")))

    simulators = {"stim", "pymatching", "ldpc", "chromobius", "panqec", "qldpc"}
    assert not (set(required) & simulators), \
        "the measurement stack is a required dependency again: %s" % sorted(set(required) & simulators)


def test_the_measure_extra_covers_the_rest_of_the_full_stack():
    """required + [measure] reconstructs requirements.txt, so no tier can drift."""
    required, extras = _pyproject_deps()
    assert extras["measure"], "no [project.optional-dependencies] measure array"
    assert set(extras["dev"]) >= {"pytest", "pyflakes", "shellcheck-py"}, (
        "the [dev] extra must install the lint gates, or they skip and stop being gates: %s"
        % extras["dev"])
    combined = set(required) | set(extras["measure"])
    full = set(_requirements("requirements.txt")) - {"pytest"}
    assert combined == full, (
        "pyproject tiers and requirements.txt disagree:\n"
        "  only in requirements.txt: %s\n  only in pyproject:        %s"
        % (sorted(full - combined), sorted(combined - full)))


def test_citation_and_package_metadata_name_the_same_work():
    """CITATION.cff and pyproject agree on authors, licence and repository."""
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cff = open(os.path.join(root, "CITATION.cff")).read()
    pyp = open(os.path.join(root, "pyproject.toml")).read()

    cff_authors = [(g.strip(), f.strip()) for f, g in
                   re.findall(r"family-names:\s*(.+)\n\s*given-names:\s*(.+)", cff)]
    assert cff_authors, "CITATION.cff lists no authors"
    surnames = {f for _, f in cff_authors}
    for name in re.findall(r'\{\s*name\s*=\s*"([^"]+)"', pyp):
        assert name.split()[-1] in surnames, \
            "%s is an author in pyproject but not in CITATION.cff" % name

    assert re.search(r"^license:\s*Apache-2\.0", cff, re.M), \
        "CITATION.cff licence is not Apache-2.0"
    assert 'license = { text = "Apache-2.0" }' in pyp, "pyproject licence is not Apache-2.0"
    assert os.path.exists(os.path.join(root, "LICENSE")), "no LICENSE file"

    repo_cff = re.search(r'repository-code:\s*"([^"]+)"', cff).group(1)
    repo_pyp = re.search(r'Repository\s*=\s*"([^"]+)"', pyp).group(1)
    assert repo_cff == repo_pyp, "repository URLs differ: %s vs %s" % (repo_cff, repo_pyp)


def test_resume_treats_a_missing_record_and_a_damaged_one_differently(tmp_path):
    """A runner resumes from an absent file, and refuses to resume from a broken one.

    Every runner wrote `try: json.load(open(out)) except Exception: results = []`.
    Right for a file that does not exist yet; wrong for one that does. A
    truncated record read as EMPTY, so the runner re-measured the whole study
    from scratch and then overwrote the damaged file -- discarding whatever had
    survived in it, after hours of compute, silently.
    """
    import json
    from chameleon.records import resume_records

    absent = tmp_path / "nothing_here.json"
    assert resume_records(str(absent)) == [], "an absent record must start a fresh run"

    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps([{"spec": "surf2d:3", "tier": "anchor"}]))
    assert len(resume_records(str(partial))) == 1, "a good record must still resume"

    damaged = tmp_path / "protocol_v1_A.json"
    damaged.write_text('[{"spec": "surf2d:3"},')          # interrupted write
    with pytest.raises(ValueError) as ei:
        resume_records(str(damaged))
    assert str(damaged) in str(ei.value)


def test_no_runner_swallows_a_broken_record_or_a_broken_estimator():
    """No `except Exception` may stand in for reading a record or scoring a gain.

    xfit_gain already returns None for every legitimate censoring case, measured
    across all 2,896 protocol cells and all 332 patch rows: not one raises. So a
    broad catch around it could only ever launder a bug into a None gain, which
    then reads as an honest censoring decision.
    """
    import ast, glob, re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for f in sorted(glob.glob(os.path.join(root, "scripts", "*.py"))
                    + glob.glob(os.path.join(root, "data_generator", "*.py"))
                    + glob.glob(os.path.join(root, "src", "chameleon", "*.py"))):
        src = open(f).read()
        rel = os.path.relpath(f, root)
        tree = ast.parse(src)
        lines = src.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            broad = node.type is None or getattr(node.type, "id", None) in ("Exception", "BaseException")
            if not broad:
                continue
            guarded = "\n".join(lines[node.lineno - 6:node.end_lineno])
            if re.search(r"json\.load\s*\(\s*open\s*\(\s*out\b", guarded):
                offenders.append("%s:%d resumes through a broad except" % (rel, node.lineno))
            if "xfit_gain" in guarded:
                offenders.append("%s:%d swallows an estimator failure" % (rel, node.lineno))
    assert not offenders, "\n".join(offenders)


def test_every_script_runs_without_pythonpath_preset():
    """The README's bare `python3 scripts/<name>.py` must work from a plain shell.

    Eleven of thirteen failed with `ModuleNotFoundError: No module named
    'chameleon'`. Each script carries a sys.path bootstrap for exactly this, but
    a `from chameleon...` import had drifted ABOVE it, so the bootstrap ran too
    late to matter. It was invisible in every previous check because the Makefile
    exports PYTHONPATH and this session always had it set -- the failure only
    appears from a shell that does not.
    """
    import glob, subprocess, sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["LEVER_ROOT"] = root
    broken = []
    for f in sorted(glob.glob(os.path.join(root, "scripts", "*.py"))):
        r = subprocess.run([sys.executable, f, "--help"], cwd=root, env=env,
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            broken.append("%s: %s" % (os.path.basename(f),
                                      (r.stderr.strip().splitlines() or [""])[-1][:90]))
    assert not broken, "scripts that cannot run without PYTHONPATH:\n" + "\n".join(broken)


def test_the_path_bootstrap_precedes_every_package_import():
    """Statically: no script may import chameleon before its sys.path bootstrap."""
    import ast, glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    for f in sorted(glob.glob(os.path.join(root, "scripts", "*.py"))):
        src = open(f).read()
        tree = ast.parse(src)
        lines = src.splitlines()
        boot = next((n.lineno for n in tree.body
                     if "sys.path.insert" in "\n".join(lines[n.lineno - 1:n.end_lineno])), None)
        assert boot is not None, "%s has no sys.path bootstrap" % os.path.basename(f)
        for n in tree.body:
            mod = n.module if isinstance(n, ast.ImportFrom) else None
            names = [a.name for a in n.names] if isinstance(n, ast.Import) else []
            touches = (mod or "").startswith("chameleon") or any(
                a.startswith("chameleon") for a in names)
            if touches and n.lineno < boot:
                bad.append("%s:%d imports chameleon before the bootstrap at line %d"
                           % (os.path.basename(f), n.lineno, boot))
    assert not bad, "\n".join(bad)


def test_every_runner_caps_its_worker_threads_the_same_way():
    """One shared cap, not six copies and twelve omissions.

    Six runners pinned the BLAS/OpenMP variables to 2, one to 8, and twelve did
    not pin at all -- so on a many-core host each pool worker sized its thread
    pool to the whole machine. Measured first, so this is not mistaken for a
    correctness fix: at 1 thread and at 16, frame selection and surrogate values
    for surf2d:5, color2d:3 and BB18 are bit-identical. The defect is
    oversubscription and inconsistency, not wrong numbers.
    """
    import glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    missing, handrolled = [], []
    for f in sorted(glob.glob(os.path.join(root, "scripts", "*.py"))):
        src = open(f).read()
        name = os.path.basename(f)
        if "cap_threads(" not in src:
            missing.append(name)
        if "OMP_NUM_THREADS" in src:
            handrolled.append(name)
    assert not missing, "runners that never cap worker threads: %s" % missing
    assert not handrolled, "runners still setting the variables by hand: %s" % handrolled

    from chameleon._env import cap_threads, DEFAULT_THREADS
    import os as _os
    saved = {k: _os.environ.pop(k, None) for k in
             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "CHAM_THREADS")}
    try:
        assert cap_threads() == DEFAULT_THREADS
        assert _os.environ["OMP_NUM_THREADS"] == str(DEFAULT_THREADS)
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS"):
            _os.environ.pop(k, None)
        _os.environ["CHAM_THREADS"] = "5"
        assert cap_threads() == 5, "CHAM_THREADS is ignored"
        assert _os.environ["MKL_NUM_THREADS"] == "5"
    finally:
        for k, v in saved.items():
            _os.environ.pop(k, None)
            if v is not None:
                _os.environ[k] = v


def test_no_shipped_code_selects_over_a_pool_containing_the_baselines():
    """Nothing in the artifact may pick the best of a candidate set that includes a baseline.

    The vendor module shipped a dead `m_ours` that did exactly that: its
    candidate list opened with m_css, m_xzzx, m_zxxz and m_tiurev, and it
    returned whichever scored best. Unreachable from any entry point, but a
    reviewer reading the source would reasonably conclude the method is chosen
    with the baselines in the pool -- which is the comparison this work argues
    against. It is removed; this fails if such a pool is ever built again.
    """
    import ast, glob
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BASELINE_FNS = ("m_css", "m_xzzx", "m_zxxz", "m_tiurev", "m_tiurev6")
    offenders = []
    for f in sorted(glob.glob(os.path.join(root, "src", "chameleon", "**", "*.py"), recursive=True)
                    + glob.glob(os.path.join(root, "scripts", "*.py"))):
        src = open(f).read()
        rel = os.path.relpath(f, root)
        for node in ast.walk(ast.parse(src)):
            if not isinstance(node, (ast.List, ast.Tuple)):
                continue
            called = {getattr(e.func, "id", getattr(getattr(e.func, "attr", None), "lower", lambda: None)())
                      for e in node.elts if isinstance(e, ast.Call)}
            named = {getattr(e, "id", None) for e in node.elts if isinstance(e, ast.Name)}
            hits = (called | named) & set(BASELINE_FNS)
            if len(hits) >= 2:
                offenders.append("%s:%d builds a candidate pool from baselines %s"
                                 % (rel, node.lineno, sorted(h for h in hits if h)))
    assert not offenders, "\n".join(offenders)


def test_vendor_modules_expose_only_what_the_artifact_uses():
    """The vendor modules are constructions and baselines -- not experiment drivers.

    Both shipped a standalone `main`/`cell`/method-table driver that no entry
    point reached, complete with env-var globals, a BP+OSD decoder, a device
    noise sampler, and a `json.dump` into `results/` through a RELATIVE path --
    so running one from the artifact root would write into the shipped record
    tree. Removing them cut the two modules from 216 lines to under 130 and left
    every construction and baseline frame bit-identical (checked against the
    pre-removal module: 25 of 25 frames and code matrices).
    """
    from chameleon.vendor import multicode_deform as MC, qldpc_deform as Q
    for name in ("get_code", "cb", "m_xzzx", "m_zxxz", "m_tiurev", "m_tiurev6", "CODES"):
        assert hasattr(MC, name), "multicode_deform lost %s" % name
    for name in ("get_bb", "SPECS"):
        assert hasattr(Q, name), "qldpc_deform lost %s" % name
    for name in ("m_ours", "METHODS", "main", "cell", "ler", "chan", "field", "device_pool"):
        assert not hasattr(MC, name), "the legacy driver is back in multicode_deform: %s" % name
    for name in ("main", "cell"):
        assert not hasattr(Q, name), "the legacy driver is back in qldpc_deform: %s" % name


def test_nothing_writes_into_results_through_a_relative_path():
    """A write to `results/...` must go through rpath, or it lands wherever you ran from."""
    import glob, re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bad = []
    for f in sorted(glob.glob(os.path.join(root, "src", "chameleon", "**", "*.py"), recursive=True)
                    + glob.glob(os.path.join(root, "scripts", "*.py"))
                    + glob.glob(os.path.join(root, "data_generator", "*.py"))):
        for i, line in enumerate(open(f), 1):
            if re.search(r'open\(\s*"results/', line) or re.search(r"open\(\s*'results/", line):
                bad.append("%s:%d writes to a cwd-relative results/ path" % (os.path.relpath(f, root), i))
    assert not bad, "\n".join(bad)


def test_value_emission_is_organised_into_named_sections():
    """paper_values.main dispatches to named sections; it does not compute inline.

    main() was 341 lines: a dozen tidy `section(...)` calls, and then ~180 lines
    of inline computation -- a 64-line per-cell loop and six side-artifact blocks
    -- that had never been extracted. This is the code that produces every number
    quoted in the paper, so it is the code most likely to be read closely, and a
    reader could not see which block produced which macro.
    """
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tree = ast.parse(open(os.path.join(root, "data_generator", "paper_values.py")).read())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    length = main.end_lineno - main.lineno + 1
    assert length <= 160, "main() has grown back to %d lines; extract the new work into a " \
                          "_values_* function like its neighbours" % length

    sections = [n.name for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("_values_")]
    assert len(sections) >= 18, "expected the value sections to stay extracted, found %d" % len(sections)

    # every extracted section must actually be dispatched from main
    called = {n.func.id for n in ast.walk(main) if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Name)}
    orphans = [s for s in sections if s not in called]
    assert not orphans, "value sections defined but never called: %s" % orphans


# ---------------------------------------------------------------- baseline frames
# The four fixed baselines are what Chameleon is measured against, so they define
# every reported gain. Nothing pinned them: a mutation test changed m_tiurev's
# checkerboard phase and the byte-identity gate, the whole pytest suite and smoke
# ALL PASSED -- because the exported tables are rebuilt from the stored RECORDS,
# so they cannot see the measurement path at all. A silently altered baseline
# would quietly redefine the comparison in every future run.
# Frames as measured on the shipped calibration (berlin_star, anchor p, seed 0).

GOLDEN_BASELINE_FRAMES = {
    'surf2d:3': {
        'CSS'    : "000000000",
        'XZZX'   : "010101010",
        'ZXXZ'   : "101010101",
        'Tiurev' : "011101100",
        'Tiurev6': "103250341",
    },
    'surf2d:5': {
        'CSS'    : "0000000000000000000000000",
        'XZZX'   : "0101010101010101010101010",
        'ZXXZ'   : "1010101010101010101010101",
        'Tiurev' : "0111011001001011010101000",
        'Tiurev6': "1032503410134442101210135",
    },
    'color2d:3': {
        'CSS'    : "0000000",
        'XZZX'   : "1101110",
        'ZXXZ'   : "0010001",
        'Tiurev' : "1111111",
        'Tiurev6': "1045504",
    },
    'BB18': {
        'CSS'    : "000000000000000000",
        'XZZX'   : "111111111000000000",
        'ZXXZ'   : "000000000111111111",
        'Tiurev' : "110111001001111000",
        'Tiurev6': "113551331003343200",
    },
}


def test_baseline_frames_are_exactly_what_the_measurements_used():
    """Each fixed baseline reproduces its golden frame, bit for bit."""
    from chameleon.core import Code, NoiseField, baseline_frames
    from chameleon.vendor.multicode_deform import m_tiurev6
    from chameleon.config import ProtocolConfig, BASELINES

    cfg = ProtocolConfig.default()
    for spec, want in GOLDEN_BASELINE_FRAMES.items():
        code = Code(spec)
        field = NoiseField("berlin_star", cfg.p_anchor, 0, code)
        got = baseline_frames(code, field)
        assert set(got) == set(BASELINES), "the baseline set changed: %s" % sorted(got)
        for name in BASELINES:
            actual = "".join(map(str, got[name].tolist()))
            assert actual == want[name], (
                "%s baseline on %s changed:\n  was %s\n  now %s"
                % (name, spec, want[name], actual))
        six = "".join(map(str, m_tiurev6(code.C, field.pX, field.pY, field.pZ)))
        assert six == want["Tiurev6"], (
            "the six-frame Tiurev rule on %s changed:\n  was %s\n  now %s"
            % (spec, want["Tiurev6"], six))


def test_the_baselines_are_actually_fixed_rules():
    """XZZX and ZXXZ ignore the noise; Tiurev reads only the per-qubit ordering.

    If a baseline started adapting to the noise map it would stop being the fixed
    comparison the paper describes, and every reported gain would mean something
    else. Checked by handing them a completely different noise map.
    """
    from chameleon.core import Code, NoiseField, baseline_frames
    from chameleon.config import ProtocolConfig

    cfg = ProtocolConfig.default()
    code = Code("surf2d:5")
    a = baseline_frames(code, NoiseField("berlin_star", cfg.p_anchor, 0, code))
    b = baseline_frames(code, NoiseField("willow_star", cfg.p_anchor, 3, code))
    for name in ("CSS", "XZZX", "ZXXZ"):
        assert (a[name].perm == b[name].perm).all(), \
            "%s changed with the noise map; it is supposed to be noise-independent" % name
    assert not (a["Tiurev"].perm == b["Tiurev"].perm).all(), \
        "Tiurev did not react to a different noise map at all -- it reads the per-qubit " \
        "dominant axis, so some qubit should differ"


# ---------------------------------------------------------------- measurement path
# The exported tables are rebuilt from the stored RECORDS, so the byte-identity
# check can never observe the code that PRODUCES a measurement. A mutation run
# confirmed how wide that blind spot was: perturbing gamma, the deploy margin,
# the CEM seed, the chunk-seed schedule, the anchor event threshold, and even
# worst_axis -> best_axis all left every one of 87 tests passing.
#
# The tests that looked like they covered this checked PROPERTIES, not behaviour:
# an injective-but-different seed schedule is still injective, and a config
# forwarded correctly is still forwarded when max becomes min. These pin the
# behaviour itself.

GOLDEN_DEPLOYED_FRAMES = {
    ('surf2d:3', 'berlin_star', 0): "110101111",
    ('surf2d:5', 'willow_star', 1): "0100010000100111001100010",
    ('color2d:3', 'miami_star', 0): "0111101",
    ('BB18', 'berlin_star', 2): "100100000001011110",
    ('surf2d:3', 'xyz:10', 0): "133000110",
}

GOLDEN_CHUNK_SEEDS = [
    (0, 0, 'z', 1100),
    (0, 0, 'x', 1600),
    (0, 1, 'z', 1117),
    (1, 0, 'z', 2100),
    (3, 7, 'x', 4719),
    (12, 79, 'z', 14443),
]


def test_worst_axis_reports_the_worse_basis_not_the_better_one():
    """max(fX, fZ), not min -- reporting the better basis would inflate every gain.

    A frame is only as good as the axis it protects least; scoring the better
    one is the single most flattering change that could be made to the pipeline,
    and nothing detected it.
    """
    import chameleon.core as core
    from chameleon.config import ProtocolConfig, TierConfig

    # 3 failures on the X-basis chunks, 1 on the Z-basis chunks
    def fake_decode_chunk(args):
        mem = args[5]
        return (3 if mem == "x" else 1), args[7]

    class SerialPool:
        def map(self, f, xs):
            return list(map(f, xs))

    cfg = ProtocolConfig()
    cfg.sampling.nch = 1
    cfg.sampling.per = 100
    cfg.tiers["anchor"] = TierConfig(minev=0, cap=1)
    import unittest.mock as _mock
    code = core.Code("surf2d:3")
    field = core.NoiseField("xyz:10", 0.005, 0, code)
    with _mock.patch.object(core, "_decode_chunk", fake_decode_chunk):
        res = core.PhenoEvaluator(cfg, SerialPool()).worst_axis(
            code, field, core.Frame.identity(9), tier="anchor")
    assert (res.ev_x, res.ev_z) == (3, 1), (res.ev_x, res.ev_z)
    assert res.events == 3, "worst_axis reported %d events; the worse basis had 3" % res.events
    assert res.ler == 3 / res.shots, "the reported LER is not the worse basis's"


def test_a_tier_samples_until_its_event_threshold():
    """The stopping rule is the tier's minev; lowering it must change when a cell stops."""
    import chameleon.core as core
    from chameleon.config import ProtocolConfig, TierConfig

    waves = {"n": 0}

    def fake_decode_chunk(args):
        return 2, args[7]                      # 2 failures per chunk per basis

    class CountingPool:
        def map(self, f, xs):
            waves["n"] += 1
            return list(map(f, xs))

    cfg = ProtocolConfig()
    cfg.sampling.nch = 1
    cfg.sampling.per = 10
    cfg.tiers["anchor"] = TierConfig(minev=7, cap=10 ** 9)
    import unittest.mock as _mock
    code = core.Code("surf2d:3")
    field = core.NoiseField("xyz:10", 0.005, 0, code)
    with _mock.patch.object(core, "_decode_chunk", fake_decode_chunk):
        res = core.PhenoEvaluator(cfg, CountingPool()).worst_axis(
            code, field, core.Frame.identity(9), tier="anchor")
    # 2 events per wave per basis -> the threshold of 7 is first met in wave 4
    assert waves["n"] == 4, "stopped after %d waves; minev=7 at 2 events/wave needs 4" % waves["n"]
    assert res.events >= 7 and not res.unresolved


def test_the_chunk_seed_schedule_is_exactly_the_shipped_one():
    """Injectivity is not enough: the shipped records were measured on THESE seeds."""
    from chameleon.config import SamplingConfig
    s = SamplingConfig()
    for wave, chunk, mem, want in GOLDEN_CHUNK_SEEDS:
        got = s.chunk_seed(wave, chunk, mem)
        assert got == want, ("chunk_seed(%d, %d, %r) is %d, was %d -- the sampling schedule "
                             "changed, so a re-measurement no longer reproduces the shipped "
                             "records" % (wave, chunk, mem, got, want))


def test_the_selector_reproduces_its_deployed_frames():
    """The frame the protocol deploys, pinned end to end on fixed inputs.

    This is the method itself: the gamma surrogate, the binary warm start, the
    S3 CEM, the Y-slot tie-break and the deploy margin, all in one number. Any
    of them drifting changes what Chameleon IS, and none of them was pinned.
    """
    from chameleon.core import Code, NoiseField, BlindS3Selector
    from chameleon.config import ProtocolConfig

    cfg = ProtocolConfig.default()
    for (spec, noise, seed), want in GOLDEN_DEPLOYED_FRAMES.items():
        code = Code(spec)
        field = NoiseField(noise, cfg.p_anchor, seed, code)
        got = "".join(map(str, BlindS3Selector(cfg.selector).select(code, field).tolist()))
        assert got == want, ("the deployed frame for %s on %s seed %d changed:\n"
                             "  was %s\n  now %s" % (spec, noise, seed, want, got))


def test_deployed_tier_budgets_are_the_shipped_ones():
    """Each tier's event threshold and shot cap, pinned to what the records were measured on.

    A mutation run changed the anchor threshold from 100 events to 50 and every
    test still passed. The threshold decides when a cell is RESOLVED, which
    decides which cells report a gain at all (v1.9.4 censoring) -- so halving it
    silently redefines the evidence bar for every future measurement while the
    shipped records keep the old one. The caps are pinned for the same reason:
    they set how far a hard cell is chased before it is called unresolved.
    """
    from chameleon.config import ProtocolConfig
    from chameleon.core import Matrix

    want = {"anchor": (100, 12_000_000),
            "dist":   (30,  2_000_000),
            "eta":    (60,  4_000_000),
            "psweep": (60,  4_000_000)}
    tiers = ProtocolConfig.default().tiers
    assert set(tiers) == set(want) == set(Matrix.TIERS), (
        "the tier set changed: config has %s, Matrix.TIERS has %s"
        % (sorted(tiers), sorted(Matrix.TIERS)))
    for name, (minev, cap) in want.items():
        assert (tiers[name].minev, tiers[name].cap) == (minev, cap), (
            "tier %r budget changed: minev/cap were %d/%d, now %d/%d"
            % (name, minev, cap, tiers[name].minev, tiers[name].cap))


def test_the_protocol_operating_point_is_the_documented_one():
    """The anchor rate and the p grid stay inside the protocol's own hard limit.

    p = 0.005 is the anchor every headline number is quoted at, and no rate in
    the artifact may exceed 0.03 -- above that the phenomenological model is not
    the regime the paper describes.
    """
    from chameleon.config import ProtocolConfig
    cfg = ProtocolConfig.default()
    assert cfg.p_anchor == 0.005, "the anchor rate moved to %g" % cfg.p_anchor
    assert cfg.p_anchor in cfg.p_grid, "the anchor is not on the swept grid"
    assert max(cfg.p_grid) <= 0.03, "p grid exceeds the 0.03 ceiling: %s" % (cfg.p_grid,)
    assert cfg.q_ratio == 1.0, "the measurement-to-data ratio moved to %g" % cfg.q_ratio
    assert cfg.model == "pheno-qp", cfg.model


# ------------------------------------------------------- construction-level pins
# A second mutation round reached the parts the earlier pins did not: the noise
# field construction, the ambiguity enumeration depth, and the frame algebra.

GOLDEN_DEVICE_FIELD = {
    # device -> (mean data total rate, mean ancilla total rate, first data qubit's pX)
    # on surf2d:5 (25 data, 24 ancilla), seed 0, p = 0.005
    'berlin_star': (5.659820652469e-03, 4.312686820344e-03, 1.310791890110e-03),
    'miami_star':  (5.226648613112e-03, 4.763907694675e-03, 5.969102598303e-04),
    'willow_star': (3.718265682505e-03, 6.335139914058e-03, 8.023771273572e-04),
}


def test_device_noise_fields_are_exactly_the_shipped_construction():
    """A device field's data and ancilla rates, pinned to what was measured.

    The construction rescales twice: once inside field3, once jointly across data
    and ancilla. Swapping the inner mean for a median is deliberately NOT caught
    here, and the reason was measured rather than assumed: the outer joint
    rescale cancels the inner scalar, leaving a difference of one ULP in the
    ancilla mean (4.31268682034444042e-03 vs ...129e-03, relative 2e-16). A
    tolerance tight enough to see that would fail on a different BLAS or numpy
    build, and this artifact is verified to reproduce under the newest releases.
    What this pins is a real change to the construction -- the pool, the seed
    derivation, the data/ancilla split, the operating point.
    """
    import numpy as np
    from chameleon.fields import fields_ext

    for dev, (want_data, want_anc, want_p0) in GOLDEN_DEVICE_FIELD.items():
        (pX, pY, pZ), (aX, aY, aZ) = fields_ext(dev, 25, 24, 0, 0.005)
        data = float((pX + pY + pZ).mean())
        anc = float((aX + aY + aZ).mean())
        assert abs(data - want_data) < 1e-15, "%s data mean rate moved: %.12e vs %.12e" % (
            dev, data, want_data)
        assert abs(anc - want_anc) < 1e-15, "%s ancilla mean rate moved: %.12e vs %.12e" % (
            dev, anc, want_anc)
        assert abs(float(pX[0]) - want_p0) < 1e-15, "%s per-qubit rates moved" % dev
        # the joint rescale still holds the combined operating point
        allq = np.concatenate([pX + pY + pZ, aX + aY + aZ])
        assert abs(allq.mean() - 0.005) < 1e-12, dev


def test_ambiguity_enumeration_runs_to_its_saturating_depth():
    """The default enumeration depth is w_min+2, and that depth is saturated.

    logops' default `extra` is Step 1 of the method -- how deep the ambiguity
    operators are enumerated. Dropping it to 1 changed nothing that any test
    could see. Pinned here together with the evidence for the choice: on
    surf2d:3 the set grows 8 -> 13 from extra=1 to extra=2 and then stops, so 2
    is where it converges and 1 genuinely under-enumerates.
    """
    import inspect
    from chameleon.mechs import logops
    from chameleon.codes import get_code

    assert inspect.signature(logops).parameters["extra"].default == 2, \
        "the default enumeration depth changed; the shipped mechanism sets were built at 2"

    C, _, _ = get_code("surf2d:3")
    counts = {e: (len(logops(C["Hz"], C["LzZ"], C["n"], extra=e)),
                  len(logops(C["Hx"], C["LxX"], C["n"], extra=e))) for e in (1, 2, 3)}
    assert counts[1] == (8, 8), counts
    assert counts[2] == (13, 13), counts
    assert counts[3] == counts[2], "the set is no longer saturated at w_min+2: %s" % counts


def test_the_frame_table_is_the_frame_algebra():
    """PERMS defines what each frame index MEANS; conj1 must agree with it.

    A frame is stored and reported as an index into this table, so reordering it
    silently reinterprets every frame in the artifact. Nothing caught a swap of
    the two 3-cycles -- the deployed selector happens never to choose them (over
    20 cells it emits only 0-3) and no shipped record stores a frame, so the
    change is invisible through behaviour alone. It is pinned directly instead.
    """
    from chameleon.fields import PERMS
    from chameleon.pheno import conj1

    assert PERMS == [(0, 1, 2), (2, 1, 0), (1, 0, 2), (0, 2, 1), (1, 2, 0), (2, 0, 1)], \
        "the frame table was reordered: every stored frame index now means something else"
    assert PERMS[0] == (0, 1, 2), "frame 0 must be the identity"
    assert PERMS[1] == (2, 1, 0), "frame 1 must be the X<->Z swap (the binary {I,H} frame)"
    assert len(set(PERMS)) == 6 and all(sorted(p) == [0, 1, 2] for p in PERMS), \
        "PERMS is not the six permutations of (X, Y, Z)"

    # conj1 presents (pX, pY, pZ) through a frame; it must read the same table
    rates = [0.1, 0.2, 0.3]
    for i, perm in enumerate(PERMS):
        assert conj1(rates, i) == [rates[perm[0]], rates[perm[1]], rates[perm[2]]], \
            "conj1 disagrees with PERMS at frame %d" % i
    # rates are relabelled, never created or destroyed
    for i in range(6):
        assert abs(sum(conj1(rates, i)) - sum(rates)) < 1e-15


def test_the_config_fingerprint_distinguishes_configurations():
    """cfg_id must change when any knob changes -- including the name.

    Two shipped revisions carry the SAME md5 under different names
    (`v1.9:edf9f889` and `v1.9.5:edf9f889`), which the current formula cannot
    produce: the name is part of the serialised config it hashes. Those records
    predate the formula. This pins that the formula in use now really does
    separate configurations, so the fingerprint means what the README says.
    """
    import hashlib
    from chameleon.config import ProtocolConfig

    def fingerprint(cfg):
        return "%s:%s" % (cfg.name, hashlib.md5(cfg.to_json().encode()).hexdigest()[:8])

    base = ProtocolConfig.default()
    assert fingerprint(base) == fingerprint(ProtocolConfig.default()), "not deterministic"

    renamed = ProtocolConfig.default(); renamed.name = base.name + ".test"
    assert fingerprint(renamed).split(":")[1] != fingerprint(base).split(":")[1], \
        "the hash ignores the config name, so two revisions can share a fingerprint"

    for field, value in (("p_anchor", 0.01), ("q_ratio", 0.5)):
        v = ProtocolConfig.default(); setattr(v, field, value)
        assert fingerprint(v) != fingerprint(base), "changing %s did not change cfg_id" % field
    v = ProtocolConfig.default(); v.selector.cem_seed = 99
    assert fingerprint(v) != fingerprint(base), "a nested knob does not reach the fingerprint"
