#!/usr/bin/env bash
# Stage 2 -- turn the shipped records into results.
#
# Emits data, not LaTeX:
#   output/data/*.csv               one tidy CSV per result family
#   output/chameleon_results.xlsx   the same tables, one sheet each
#
#   bash scripts/reproduce_results.sh          # everything
#   bash scripts/reproduce_results.sh values   # the paper's reported numbers only
#   bash scripts/reproduce_results.sh data     # the measurement export only
set -euo pipefail

cd "$(dirname "$0")/.."
export LEVER_ROOT="${LEVER_ROOT:-$PWD}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export CHAM_PAPER_DIR="${CHAM_PAPER_DIR:-output}"

WHAT="${1:-all}"
mkdir -p "$CHAM_PAPER_DIR"/data

# paper_values.csv is written first: the export reports it as one of its tables.
if [ "$WHAT" = all ] || [ "$WHAT" = values ]; then
    echo ">> the paper's reported numbers"
    python3 data_generator/paper_values.py
fi

if [ "$WHAT" = all ] || [ "$WHAT" = data ]; then
    echo ">> measurement export"
    python3 data_generator/export_results.py
fi

echo
echo "REPRODUCE-RESULTS: DONE -> $CHAM_PAPER_DIR/{data,chameleon_results.xlsx}"
