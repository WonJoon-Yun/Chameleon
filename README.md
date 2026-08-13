# Chameleon: Computationally Efficient Per-Qubit Clifford Deformation for Non-uniform Biased Noise

Chameleon is a compiler that optimizes Clifford deformation with our surrogate model. 
It takes a stabilizer code and a calibrated per-qubit Pauli noise map, and selects a per-qubit Clifford frame out of the full 6^n space that lowers the logical error rate (LER) — with no extra qubits, no extra syndrome rounds, no code-distance cost, and **no decoder in the optimization loop**.


## Contents

```
Chameleon/
├── README.md                 this file
├── LICENSE                   Apache-2.0
├── CITATION.cff              citation metadata
├── Makefile                  every entry point
├── setup.sh                  virtualenv or conda environment
├── environment.yml           conda specification
├── requirements-minimal.txt  what `make all` needs: numpy, pandas, xlsx
├── requirements.txt          full stack, lower bounds
├── requirements-lock.txt     exact versions behind the paper's numbers
├── pyproject.toml            installable package metadata
│
├── src/chameleon/            the pass and its evaluation stack
│   ├── config.py             single source of truth for every knob and seed
│   ├── codes.py              surface / Color666 / bivariate-bicycle constructions
│   ├── mechs.py              ambiguity-operator enumeration (Step 1)
│   ├── surrogate.py          Bhattacharyya gamma-product objective (Step 2)
│   ├── search.py             cross-entropy search over the binary and 6^n spaces
│   ├── baselines.py          the fixed frames: CSS, XZZX, ZXXZ, Tiurev local rule
│   ├── fields.py             per-qubit noise fields from calibration snapshots
│   ├── records.py            record loading + analysis-time gain recomputation
│   ├── estimators.py         LER and cross-fitted gain estimators
│   └── pheno.py, core.py     phenomenological circuits and the protocol matrix
│
├── scripts/                  Stage 2 — re-measure from scratch (writes results/)
│   ├── smoke.sh              kick the tires
│   ├── reproduce_all.sh      full re-measurement driver
│   ├── reproduce_results.sh  Stage-1 driver
│   ├── run_protocol.py       the protocol matrix (groups A/B/C, resumable)
│   ├── exp_*.py, *_sweep.py  supporting studies
│   └── e_column.py           table-cell formatting logic, pinned by tests/
│
├── data_generator/           Stage 1 — turn records into tables
│   ├── export_results.py     CSV + XLSX export of every measurement
│   └── paper_values.py       recompute the paper's reported numbers
│
├── results/                  shipped result records (JSON), consumed by Stage 1
├── calibration/              IBM Berlin / IBM Miami / Google Willow snapshots
├── tests/                    pytest suite incl. golden-hash circuit regression
├── docker/                   multi-architecture container (see docker/README.md)
└── output/                   generated CSV data and the XLSX workbook
```

---

## Requirements

| | |
|---|---|
| OS | Linux (Ubuntu 22.04 tested), macOS (Intel and Apple silicon) |
| Architecture | ARM64 (built and tested natively); x86-64 (built and tested under emulation, not on native x86-64 hardware) |
| Python | >= 3.9 (3.12 used for the paper) |
| Dependencies | numpy + pandas + xlsx to reproduce; the simulator stack only to re-measure |
| RAM | < 500 MB for Stages 0–1 (measured peak 352 MB); 32 GB+ for a full re-measurement |
| Disk | ~55 MB checked out; ~1 GB with the full simulator stack installed |
| GPU | not used |

---

## Quick start

Reproducing every reported result needs **numpy, pandas and an xlsx writer —
nothing else**. The simulators and decoders are only for re-measuring from
scratch, and `chameleon` loads its submodules lazily so they are never imported
on the reproduction path.

```bash
bash setup.sh --minimal   # numpy + pandas + xlsx  (seconds)
source venv/bin/activate
make all                  # ~15 s — every measured result
```

For the full stack (test suite, live cells, `make experiments`):

```bash
bash setup.sh          # ./venv with the exact pinned versions
source venv/bin/activate
make smoke             # ~2 min  — verifies the install end to end
make all               # ~15 s   — every measured result
```

Or with the container, which needs nothing preinstalled:

```bash
bash docker/build.sh          # native build for the current architecture
bash docker/run.sh make smoke
bash docker/run.sh make all   # results land in ./output on the host
```

---

## Reproduction stages

### Stage 0 — kick the tires (~2 min, any machine)

```bash
make smoke
```

1. resolves every dependency and imports the package;
2. runs the pytest suite;
3. measures one miniature pipeline cell live (surface d=3, one Willow map,
   reduced budget) and asserts the selected frame beats the undeformed code on
   the surrogate;
4. exports the measured results and checks the cell table is non-empty.

Each step prints PASS/FAIL; the script exits non-zero on any failure. On a
minimal install steps 2 and 3 report SKIP — they exercise the measurement
stack, which the reproduction path does not need.

### Stage 1 — regenerate every result (~15 s, any machine)

```bash
make values    # recompute the paper's reported numbers -> paper_values.csv
make data      # CSV + XLSX export of every measurement
make all       # both
```

Reads only `results/` and writes to `output/`. Run `make all` rather than
`make data` alone the first time: `paper_values.csv` is produced by the values
stage and reported by the export.

### Stage 2 — full re-measurement (200+ core-days, optional)

```bash
make quick          # anchor tier only, ~4 core-days
make experiments    # everything, ~40 core-days
PROCS=32 make experiments  # If you have more processors, change the value PROCS=<N_PROCS>
```

Re-runs the protocol matrix and every supporting study from scratch. Runners
checkpoint after each cell and resume by cell key, so they can be interrupted
freely and re-running skips completed cells.

Re-running a study never overwrites a shipped record. The files under
`results/` are the evidence the reported numbers were computed from, so a
re-measurement is written to `<name>.rerun.json` beside the original and the
path taken is printed. Set `CHAM_OVERWRITE_RECORDS=1` to replace them in place.

Records are resumable only within one configuration: a cell measured under a
different config is a different experiment, not a partial run of this one. The
shipped records were produced under an earlier config revision, so a
re-measurement writes to a config-tagged file beside them (named in the runner's
first line) and leaves them untouched. `--strict-resume` refuses to start
instead. Stage 1 therefore keeps reading the shipped records unless you point it
at the new file. All sampling is seeded
deterministically (`src/chameleon/config.py`), so a re-run reproduces the
shipped records bit-for-bit on the same package versions.

Approximate cost: surface + color full protocol ~25 core-days; BB18/36/72/108
anchors ~15 core-days. BB144 uses a documented reduced budget (three maps,
60-event floor) because a full-budget BB144 cell costs ~10 h under product-sum
BP+OSD.

---

## What you get

`make all` writes:

```
output/
├── chameleon_results.xlsx     every table below, one sheet each, plus a README sheet
└── data/*.csv                 the tables below
```

## How the pipeline works

Chameleon runs in three steps. `src/chameleon/config.py` is the single source of
truth for every knob and seed; each result record carries a `cfg_id` fingerprint
of the configuration it was produced under.

1. **Ambiguity-set construction (offline, once per code).** An *ambiguity
   operator* is an error pattern that produces no syndrome yet still changes the
   logical result; combining one with any pattern yields another pattern with
   the same syndrome and a different logical outcome. The set depends only on
   the code, never on the noise map, so it is enumerated once and reused for
   every calibration. Small codes are enumerated exhaustively up to weight
   $W = d+2$; BB72 and larger use randomized Gaussian elimination plus the
   translation orbit. (`src/chameleon/mechs.py`)

2. **Analytic scoring.** Exact per-operator scoring compares all $2^(w-1)$
   competing pattern pairs and does not factor qubit by qubit. Chameleon
   replaces it with the Bhattacharyya bound
   $\Gamma_l(F) = \prod_{q \in l} \gamma(r^q_c(F))$, $\gamma(r) = 2*\sqrt{r(1-r)}$,
   cutting the cost from $O(2^w)$ to $O(w)$, and sums it into a class score
   $U_c(F)$. Memory minimizes $\max({U_X, U_Z})$; single-axis workloads minimize
   $U_c*$. (`src/chameleon/surrogate.py`)

3. **Search.** Cross-entropy search over the binary $\{X,Z\}^n$ space, then
   warm-started over the full $6^n$ space; the full-frame candidate deploys only
   if it improves the objective by more than tau = 20%.
   (`src/chameleon/search.py`)

## Calibration data

`calibration/` holds the IBM Berlin and IBM Miami snapshots downloaded from the
IBM Quantum platform, and a Google Willow map derived from published data. The
noise-field construction requires these snapshots: the seeds determine the
resampling, not the underlying pool. If redistribution proves not to be
permitted at publication time, the raw snapshots can be replaced by the derived
per-qubit (pX, pY, pZ) pools — a small numeric table per device — which
preserves bit-identical reproduction.

## License

Apache License 2.0 — see `LICENSE`. Please cite the paper (`CITATION.cff`) if
you use this artifact.

## Contribution

We are open for contributors, please feel free to raise pull request.
