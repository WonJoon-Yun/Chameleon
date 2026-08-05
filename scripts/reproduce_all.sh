#!/usr/bin/env bash
# Stage 1 -- re-measure the protocol matrix from scratch.
#
#   bash scripts/reproduce_all.sh                  # everything (core-days)
#   MODE=quick bash scripts/reproduce_all.sh       # anchor tier only (~4 core-days)
#   PROCS=32 bash scripts/reproduce_all.sh         # cap worker processes
#   CHAM_GROUPS="A" bash scripts/reproduce_all.sh  # surface + color only
#
# Every runner checkpoints after each cell and resumes by cell key, so the
# script is idempotent: re-running skips cells whose records already exist.
# Sampling is seeded deterministically (src/chameleon/config.py), so a re-run
# reproduces the shipped records bit-for-bit on the same package versions.
set -euo pipefail

cd "$(dirname "$0")/.."
export LEVER_ROOT="${LEVER_ROOT:-$PWD}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

PROCS="${PROCS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 8)}"
# NB: not GROUPS -- that is a bash builtin holding the caller's group IDs, so
# assigning to it silently does nothing and the loop below would iterate over
# numeric GIDs instead of A B C.
CHAM_GROUPS="${CHAM_GROUPS:-A B C}"
MODE="${MODE:-full}"

# An array, not a string: "$TIER" would pass "--tier anchor" as a single argv
# entry, and $TIER unquoted would word-split any path in it. Neither is needed
# with "${TIER[@]}".
TIER=()
[ "$MODE" = quick ] && TIER=(--tier anchor)

echo "=============================================================="
echo " Chameleon Stage 1 -- protocol matrix"
echo "   groups : $CHAM_GROUPS"
echo "   procs  : $PROCS"
echo "   mode   : $MODE${TIER[0]:+  (${TIER[*]})}"
echo "   records: results/protocol_v1/"
echo "=============================================================="

for g in $CHAM_GROUPS; do
    echo
    echo ">> protocol group $g"
    GROUP="$g" PROCS="$PROCS" python3 scripts/run_protocol.py "${TIER[@]}"
done

# A study that crashes used to be reported as "(skipped)" and the driver carried
# on to exit 0, so a broken run looked like a complete one. Failures are recorded
# and reported at the end, and the driver exits non-zero.
FAILED=""

run_study() {
    echo "   - $1"
    if ! PROCS="$PROCS" python3 "scripts/$1"; then
        echo "     FAILED: $1"
        FAILED="$FAILED $1"
    fi
}

echo
echo ">> supporting studies"
for s in cutoff_sweep.py decode_validation.py drift_study.py abstention_analysis.py \
         qratio_sweep.py binary_ablation.py restart_probe.py patch_pheno.py; do
    run_study "$s"
done

echo
echo ">> mechanism / surrogate studies (Sec. VII)"
for s in exp_m1_exact_vs_bhatt.py exp_m2_enum_convergence.py exp_m3_search_ablation.py \
         exp_m5_drift_robustness.py exp_m7_runtime_breakdown.py; do
    run_study "$s"
done

echo
if [ -n "$FAILED" ]; then
    echo "REPRODUCE-ALL: INCOMPLETE -- these studies failed:$FAILED"
    echo "  Their records were not regenerated; the shipped ones are still in place."
    exit 1
fi
echo "REPRODUCE-ALL: DONE. Now regenerate the paper artifacts:"
echo "    make all"
