#!/usr/bin/env bash
# Kick the tires (~2 minutes, any machine, < 500 MB RAM).
#
# 1. import and dependency check
# 2. the pytest suite, including the golden-hash regression that pins the exact
#    stim circuit the protocol builds
# 3. one miniature pipeline cell measured live (surface d=3, one map, reduced
#    shot budget) -- proves the search and the decoders actually run here
# 4. the tabular export produced from the shipped records
#
# Prints PASS/FAIL per step and exits non-zero if any step fails.
set -uo pipefail

cd "$(dirname "$0")/.." || { echo "cannot enter the artifact root"; exit 1; }
export LEVER_ROOT="${LEVER_ROOT:-$PWD}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CHAM_PAPER_DIR="${CHAM_PAPER_DIR:-output}"
export SOURCE_DATE_EPOCH=0

RC=0
SKIPPED=0
step() {
    printf '\n[%s] %s\n' "$1" "$2"
}
verdict() {
    if [ "$1" -eq 0 ]; then echo "  PASS"; else echo "  FAIL"; RC=1; fi
}
# A step that never ran is not a step that passed. Reporting a skip as PASS hid
# a broken environment: when qldpc failed to import for an unrelated reason the
# suite was skipped, printed PASS, and the run still ended "ALL PASS" -- so the
# golden-hash regression silently never executed. A skip keeps the exit status
# clean (a minimal install is a supported configuration) but is counted and
# named in the summary instead of being folded into the passes.
skip() {
    SKIPPED=$((SKIPPED + 1))
}

step 1/4 "dependency and import check"
python3 - <<'EOF'
import sys
missing = []
for m in ["numpy", "pandas", "chameleon.records", "chameleon.estimators", "chameleon.config"]:
    try:
        __import__(m)
    except ImportError as e:
        missing.append("%s (%s)" % (m, e))
if missing:
    sys.exit("reproduction path incomplete:\n    " + "\n    ".join(missing))
print("  reproduction path OK (numpy, pandas, chameleon.records)")

optional = []
for m in ["stim", "pymatching", "ldpc", "chromobius", "panqec", "qldpc", "scipy"]:
    try:
        __import__(m)
    except ImportError:
        optional.append(m)
if optional:
    print("  re-measurement stack absent (%s) -- `make all` still works,"
          % ", ".join(optional))
    print("  `make experiments` and the live cell below need the full install")
else:
    import chameleon.codes, chameleon.mechs, chameleon.surrogate, chameleon.search
    print("  re-measurement stack OK (simulators and decoders present)")
EOF
verdict $?

step 2/4 "test suite (golden-hash circuit regression)"
# The suite exercises the measurement stack, so it needs the full install. On a
# minimal install (reproduction path only) it is skipped rather than failed.
if python3 -c "import stim, panqec, qldpc" >/dev/null 2>&1; then
    python3 -m pytest tests/ -q 2>&1 | tail -5
    verdict "${PIPESTATUS[0]}"
else
    echo "  SKIP: the test suite needs the re-measurement stack"
    echo "  (install requirements.txt to run it; the reproduction path is unaffected)"
    skip
fi

step 3/4 "live miniature cell (surface d=3, reduced budget)"
python3 - <<'EOF'
import sys
try:
    import numpy as np
    from chameleon.codes import get_code
except ImportError as e:
    print("  SKIP: the re-measurement stack is not installed (%s)" % e)
    print("  (a minimal install reproduces every result; only this live cell needs it)")
    sys.exit(3)   # 3 = skipped, so the caller does not record it as a pass
from chameleon.mechs import mechs
from chameleon.fields import field3x
from chameleon.surrogate import build_U6
from chameleon.search import cem6
from chameleon import baselines as bl

spec, p, seed = "surf2d:3", 0.005, 3
C, _, _ = get_code(spec)
n = C["n"]
LX, LZ = mechs(spec, C)
pX, pY, pZ = field3x("willow_star", n, seed, p)
U = build_U6(LX, LZ, n, pX, pY, pZ)

css = np.zeros(n, int)
u_css = float(U(css)[0])
best = cem6(U, n, iters=6, M=80, seed=0)
u_best = float(U(np.asarray(best, int)[None])[0])
print("  code %s, n=%d, %d/%d ambiguity operators (X/Z)" % (spec, n, len(LX), len(LZ)))
print("  U(CSS)       = %.6e" % u_css)
print("  U(Chameleon) = %.6e   (%.2fx lower)" % (u_best, u_css / u_best))
assert u_best <= u_css, "search returned a worse frame than the undeformed code"
print("  search improves on the undeformed baseline")
EOF
rc=$?
if [ "$rc" -eq 3 ]; then skip; else verdict "$rc"; fi

step 4/4 "export measured results from shipped records"
python3 data_generator/export_results.py --csv-only >/dev/null 2>&1
if [ -s "$CHAM_PAPER_DIR/data/protocol_cells.csv" ]; then
    echo "  wrote $CHAM_PAPER_DIR/data/protocol_cells.csv ($(($(wc -l < "$CHAM_PAPER_DIR/data/protocol_cells.csv") - 1)) measured cells)"
    verdict 0
else
    verdict 1
fi

echo
if [ "$RC" -ne 0 ]; then
    echo "SMOKE: FAILURES ABOVE."
elif [ "$SKIPPED" -ne 0 ]; then
    echo "SMOKE: PASS, but $SKIPPED step(s) SKIPPED -- NOT a full check."
    echo "The reproduction path works; the re-measurement stack was never exercised."
    echo "Install requirements.txt and re-run to cover the skipped step(s)."
    echo "Next:  make all                 (~15 s, every measured result)"
else
    echo "SMOKE: ALL PASS -- the artifact is ready."
    echo "Next:  make all                 (~15 s, every measured result)"
fi
exit "$RC"
