#!/usr/bin/env bash
# Create (or refresh) the Chameleon artifact environment.
#
#   bash setup.sh            # virtualenv at ./venv, pinned versions (everything)
#   bash setup.sh --minimal  # only what `make all` needs: numpy + pandas + xlsx
#   bash setup.sh --conda    # conda env "chameleon-ae" from environment.yml
#   bash setup.sh --loose    # virtualenv with lower-bound requirements
#
# Works on x86-64 and ARM64 Linux and on macOS (Intel and Apple silicon).
set -euo pipefail

cd "$(dirname "$0")"

MODE=venv
REQ=requirements-lock.txt

usage() {
    sed -n '2,9p' "$0" | sed 's/^# \?//'
}

for arg in "$@"; do
    case "$arg" in
        --conda) MODE=conda ;;
        --loose) REQ=requirements.txt ;;
        --minimal) REQ=requirements-minimal.txt ;;
        -h|--help) usage; exit 0 ;;
        *)
            echo "unknown option: $arg" >&2
            echo >&2
            usage >&2
            exit 2 ;;
    esac
done

if [ "$MODE" = conda ]; then
    command -v conda >/dev/null || { echo "conda not found on PATH" >&2; exit 1; }
    if conda env list | awk '{print $1}' | grep -qx chameleon-ae; then
        echo ">> updating conda env chameleon-ae"
        conda env update -n chameleon-ae -f environment.yml --prune
    else
        echo ">> creating conda env chameleon-ae"
        conda env create -f environment.yml
    fi
    echo
    echo ">> done. Activate with:  conda activate chameleon-ae"
    exit 0
fi

PY="${PYTHON:-python3}"
"$PY" - <<'EOF'
import sys
if sys.version_info < (3, 9):
    sys.exit("Python >= 3.9 required, found %d.%d" % sys.version_info[:2])
EOF

if [ ! -d venv ]; then
    echo ">> creating virtualenv ./venv ($("$PY" --version 2>&1))"
    "$PY" -m venv venv
fi
# shellcheck disable=SC1091
source venv/bin/activate

echo ">> installing $REQ"
pip install --upgrade pip >/dev/null
pip install -r "$REQ"

echo ">> installing the chameleon package (editable)"
pip install --no-deps -e .

echo
if [ "$REQ" = requirements-minimal.txt ]; then
    python -c "import chameleon.records, numpy, pandas; print('import check OK (minimal: reproduction path only)')"
else
    python -c "import chameleon, stim, pymatching, ldpc, pandas; import chameleon.codes; print('import check OK (full)')"
fi
echo
echo ">> done. Activate with:  source venv/bin/activate"
if [ "$REQ" = requirements-minimal.txt ]; then
    echo ">> then:                 make all      (re-measurement needs the full install)"
else
    echo ">> then:                 make smoke"
fi
