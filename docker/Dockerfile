# Chameleon artifact image.
#
# Multi-architecture: builds natively on linux/amd64 (x86-64) and linux/arm64
# (AWS Graviton, Ampere, Apple silicon under Docker Desktop). Every dependency
# ships aarch64 and x86-64 wheels, so no cross-compilation is required.
#
#   docker build -t chameleon-ae -f docker/Dockerfile .
#   docker run --rm -it chameleon-ae make smoke
#
# See docker/build.sh for the buildx multi-platform recipe.

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="Chameleon artifact" \
      org.opencontainers.image.description="Per-qubit Clifford deformation for non-uniform biased noise" \
      org.opencontainers.image.licenses="Apache-2.0"

# Build toolchain: ldpc / stim / pymatching publish wheels for both
# architectures, but a source build is the fallback on unusual platforms.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git make \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /artifact

# Dependency layer first so code edits do not invalidate the wheel cache.
COPY requirements-lock.txt requirements.txt pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install -r requirements-lock.txt

COPY . .
RUN pip install --no-deps -e .

# The generators resolve every path relative to LEVER_ROOT and write data and
# tables under $CHAM_PAPER_DIR, both of which stay inside the artifact tree.
ENV PYTHONPATH=/artifact/src \
    LEVER_ROOT=/artifact \
    CHAM_PAPER_DIR=output

# Run correctly under an arbitrary uid, not just root. docker/run.sh passes
# `-u $(id -u):$(id -g)` under rootful docker, and then every default cache path
# lands somewhere the image tree does not grant that uid: numba compiles the
# galois ufuncs with cache=True and writes beside the module in site-packages
# ("cannot cache function 'int_to_vector': no locator available"), matplotlib
# falls back to $HOME=/ , and $HOME itself is unwritable. Point all three at
# /tmp, which is world-writable, so the import of qldpc -> galois -> numba
# succeeds for any uid.
ENV HOME=/tmp \
    NUMBA_CACHE_DIR=/tmp/numba-cache \
    MPLCONFIGDIR=/tmp/matplotlib

# run_protocol.py takes an exclusive lock next to the record it writes, so the
# shipped results tree has to be writable by the same arbitrary uid -- otherwise
# the lock raises PermissionError before the argument validation under test can
# run. X sets +x on directories only, leaving file modes alone.
RUN chmod -R a+rwX /artifact/results

RUN python -c "import chameleon, stim, pymatching, ldpc, pandas; print('chameleon import OK')"

CMD ["make", "smoke"]
