"""ProtocolConfig — every experimental knob as one dataclass.

Change a parameter, rerun: nothing else in the pipeline hardcodes a value.
`ProtocolConfig.default()` returns the deployed configuration; any variant
is an explicit, serializable object (`to_json`/`from_json`), so a run can always be
traced back to its full configuration.
"""
from dataclasses import dataclass, field, asdict
import json


# ---------------------------------------------------------------- canonical names
# The four fixed-frame baselines, in the order the tie-break depends on: when two
# baselines have exactly equal LER, the best-baseline gain picks the first of these.
# Every estimator, runner and exporter reads this one tuple, so the order cannot
# drift between them.
BASELINES = ("CSS", "XZZX", "ZXXZ", "Tiurev")
DEPLOYED = "Cham"                       # the frame Chameleon selects
POLICIES = BASELINES + (DEPLOYED,)      # every policy scored on a cell


@dataclass
class SamplingConfig:
    """The shot schedule: how a cell's Monte-Carlo budget is split into parallel work.

    A measurement runs in *waves*; each wave is ``nch`` chunks of ``per`` shots on
    each of the two memory bases, so one wave is ``2*nch*per`` shots and saturates
    ``2*nch`` cores. Waves repeat until the tier's event threshold is met or its
    shot cap is passed. Every chunk's seed is a pure function of its position
    (see `chunk_seed`), so a run is reproducible regardless of how the chunks are
    scheduled across workers or how many workers there are.
    """

    per: int = 20_000          # shots per chunk (kept small: BP+OSD decodes per-syndrome, so a
                               # large batch just over-samples per wave; parallelism is across chunks)
    nch: int = 80              # chunks per wave per axis (2*80=160 chunks -> saturates 160 cores)
    seed0: int = 1100          # seed = seed0 + 1000*wave + 17*chunk (+500 for X basis)

    def chunk_seed(self, wave, chunk, mem):
        """Seed for one (wave, chunk, basis) sampler.

        The strides (1000 per wave, 17 per chunk, 500 between bases) keep the
        three indices from aliasing onto a shared seed within any run this
        schedule can produce, so no two chunks sample the same shots.
        """
        return self.seed0 + 1000 * wave + 17 * chunk + (0 if mem == "z" else 500)


@dataclass
class TierConfig:
    """A tier's stopping rule: sample until ``minev`` failures, but never past ``cap`` shots.

    Reaching ``minev`` first makes the cell *resolved* -- its LER has enough events
    to carry a meaningful relative error. Hitting ``cap`` first leaves it
    unresolved: the rate sits below what the budget can measure, and the protocol
    then reports no gain for the cell rather than treating a floor as a win.
    """

    minev: int                 # resolved-cell event threshold
    cap: int                   # shots per mask per cell


@dataclass
class SelectorConfig:
    """Knobs of the selection rule: the CEM search over frames, and the gate on its output.

    The search is a cross-entropy method over the 6^n per-qubit frame space,
    warm-started from a cheaper binary {I,H} CEM. `deploy_margin` then decides
    whether the S3 result is actually deployed -- see the note on that field.
    """

    cem_iters: int = 50
    cem_pop: int = 500
    cem_elite_frac: float = 0.10
    cem_restarts: int = 3      # two cold + one warm (binary warm start)
    cem_seed: int = 5
    deploy_margin: float = 0.20  # v1.9.7 margin gate: deploy the S3 pick only if its
    # relative U-improvement over the binary warm pick exceeds this threshold.
    # Calibrated from the 12-panel scatter campaign (7,112 frame pairs after
    # excluding U-ties and LER-exact-ties): the U-better frame rank-inverts in
    # measured Type-A LER 31.4%/10.9%/0.7%/0.0% at relative gaps
    # 0-5/5-10/10-20/>20% -- below ~20% the ranking is not resolvable, so the
    # binary pick is kept (never worse than the 2^n search).
    warm_iters: int = 15       # binary warm-start CEM
    warm_pop: int = 200


@dataclass
class DecoderConfig:
    """BP+OSD settings for the qLDPC codes.

    Surface and color codes do not read this: they decode with MWPM (PyMatching)
    and Chromobius respectively, neither of which is parameterized here.
    """

    bp_iters: int = 1000
    osd_order: int = 7
    # serial-schedule product-sum BP: the only setting measured to dominate in
    # every swept regime. Parallel-schedule BP (any variant) collapses when
    # measurement-flip priors are far below data priors (q->0: 7e-3 vs 2e-4);
    # unscaled min-sum mis-decodes single-fault syndromes at p<=5e-4; normalized
    # min-sum under-decodes the deformed frame 3x at high rate. Serial product-
    # sum passes all four fixtures (BB72, 2026-07-11).
    bp_method: str = "product_sum"
    schedule: str = "serial"
    osd_method: str = "OSD_CS"
    ms_scaling_factor: float = 1.0


@dataclass
class ProtocolConfig:
    """Every experimental knob of the protocol, in one serializable object.

    Nothing downstream hardcodes a value: the codes, noise fields, rates, seeds,
    tiers, selector and decoder all come from here, so a variant is an explicit
    object rather than an edit. Runs record the config they were produced under,
    and a record measured with different settings is kept separate rather than
    silently merged.

    The directory fields are relative to the artifact root (`_root.ROOT`, which
    `LEVER_ROOT` overrides).
    """

    name: str = "v1.9"
    model: str = "pheno-qp"            # measurement flip q = q_ratio*p, uniform; T = d rounds
    q_ratio: float = 1.0
    p_anchor: float = 0.005
    p_grid: tuple = (0.001, 0.0018, 0.0032, 0.005, 0.01)
    eta_grid: tuple = (1, 2, 5, 10, 20, 50, 100)
    fields: tuple = ("berlin_star", "miami_star", "willow_star", "xyz:10")
    codes: tuple = ("surf2d:3", "surf2d:5", "surf2d:7",
                    "color2d:3", "color2d:5",
                    "BB18", "BB36", "BB72")   # the deployed paper matrix (color d7
                    # and BB108/144 retired; review MINOR-3, 2026-07-20)
    draws_anchor: tuple = (0, 1, 2, 3, 4)
    draws_dist: tuple = tuple(range(100))  # Matrix.cells excludes anchor draws (0-4) from the dist tier => 95 dist draws 5-99
    draws_sweep: tuple = (0, 1, 2)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    tiers: dict = field(default_factory=lambda: {
        "anchor": TierConfig(minev=100, cap=12_000_000),
        "dist":   TierConfig(minev=30,  cap=2_000_000),
        "eta":    TierConfig(minev=60,  cap=4_000_000),
        "psweep": TierConfig(minev=60,  cap=4_000_000)})
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    out_dir: str = "results/protocol_v1"      # protocol matrix records
    study_dir: str = "results/typeA_investigation"   # supporting-study records
    mech_dir: str = "results/mechs"                  # precomputed ambiguity-operator sets
    pool_dir: str = "results/data"                   # per-device Pauli-channel pools

    @classmethod
    def default(cls):
        """The deployed configuration: every knob at its protocol value."""
        return cls()

    def to_json(self, path=None):
        """Serialize to a JSON string, and write it to ``path`` when given.

        Round-trips through `from_json`. Emitted alongside every record set so a
        result can always be traced back to the settings that produced it.
        """
        d = asdict(self)
        d["tiers"] = {k: asdict(v) if not isinstance(v, dict) else v for k, v in self.tiers.items()}
        s = json.dumps(d, indent=1)
        if path:
            open(path, "w").write(s)
        return s

    @classmethod
    def from_json(cls, path_or_str):
        """Rebuild a config from a JSON file path or a JSON string.

        Accepts either: an argument starting with ``{`` is treated as the document
        itself, anything else as a path. Nested sections are restored to their
        dataclasses and the sequence fields back to tuples, so a round-tripped
        config compares equal to the original rather than differing by list-vs-tuple.
        """
        s = open(path_or_str).read() if not path_or_str.lstrip().startswith("{") else path_or_str
        d = json.loads(s)
        d["sampling"] = SamplingConfig(**d.get("sampling", {}))
        d["tiers"] = {k: TierConfig(**v) for k, v in d.get("tiers", {}).items()}
        d["selector"] = SelectorConfig(**d.get("selector", {}))
        d["decoder"] = DecoderConfig(**d.get("decoder", {}))
        for k in ("p_grid", "eta_grid", "fields", "codes", "draws_anchor", "draws_dist", "draws_sweep"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)
