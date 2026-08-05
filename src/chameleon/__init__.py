"""chameleon — the Chameleon pipeline as an importable package.

Layout:

  fields     calibration pools and noise-field constructors (3-class per qubit)
  codes      code constructors (surface / color 6.6.6 / BB)
  mechs      ambiguity-operator enumeration
  surrogate  gamma-product scores (binary and S3, batch-vectorized)
  search     CEM searches (binary and categorical S3), greedy, CP-SAT certificate
  baselines  fixed frames: CSS / XZZX / ZXXZ / Tiurev
  pheno      phenomenological circuits (T=d rounds) and DEM utilities
  records    result-record loading and analysis-time gain recomputation
  estimators LER and cross-fitted gain estimators
  config     ProtocolConfig dataclass -- the single source of truth for every knob
  core       object model: Code / NoiseField / Frame / Selector / Evaluator / Cell / Matrix

Submodules load lazily (PEP 562), so importing the package costs nothing beyond
the standard library and a module pulls in its own dependencies only when it is
first touched. This keeps the reproduction path light: reading result records
and exporting them uses `records`, `estimators` and `config`, which need numpy
alone, while the simulators and decoders (stim, pymatching, ldpc, chromobius,
panqec, qldpc) load only when a measurement module is actually used.
"""
import importlib as _importlib

__all__ = ["fields", "codes", "mechs", "surrogate", "search", "baselines",
           "pheno", "records", "estimators", "config", "core"]


def __getattr__(name):
    if name in __all__:
        mod = _importlib.import_module("." + name, __name__)
        globals()[name] = mod          # cache so later attribute access skips this hook
        return mod
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(list(globals()) + __all__))
