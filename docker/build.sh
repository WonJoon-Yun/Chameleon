#!/usr/bin/env bash
# Build the Chameleon artifact image.
#
#   bash docker/build.sh                 # native single-arch build (fastest)
#   bash docker/build.sh --multi         # linux/amd64 + linux/arm64 via buildx
#   bash docker/build.sh --multi --push  # ... and push to $IMAGE
#
# Architecture coverage
#   linux/amd64  Intel/AMD x86-64 servers and desktops
#   linux/arm64  AWS Graviton, Ampere, and Apple silicon (M1-M4) under
#                Docker Desktop, which runs linux/arm64 images natively
#
# Apple silicon note: build without --multi. Docker Desktop selects linux/arm64
# automatically and every dependency ships an aarch64 wheel, so the image builds
# and runs natively (no Rosetta, no QEMU).
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-chameleon-ae}"
TAG="${TAG:-latest}"
PLATFORMS="${PLATFORMS:-linux/amd64,linux/arm64}"

MULTI=0
PUSH=0
for arg in "$@"; do
    case "$arg" in
        --multi) MULTI=1 ;;
        --push)  PUSH=1 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

if [ "$MULTI" -eq 0 ]; then
    echo ">> native build: $IMAGE:$TAG ($(uname -m))"
    docker build -t "$IMAGE:$TAG" -f docker/Dockerfile .
    echo ">> done. Run:  docker run --rm -it $IMAGE:$TAG make smoke"
    exit 0
fi

if ! docker buildx version >/dev/null 2>&1; then
    echo "docker buildx is required for --multi (Docker >= 19.03)" >&2
    exit 1
fi

docker buildx inspect chameleon-builder >/dev/null 2>&1 \
    || docker buildx create --name chameleon-builder --use --bootstrap

OUT="--load"
if [ "$PUSH" -eq 1 ]; then
    OUT="--push"
elif [ "${PLATFORMS//,/}" != "$PLATFORMS" ]; then
    # --load cannot materialise a multi-platform manifest into the local store
    OUT="--output=type=oci,dest=chameleon-ae-multiarch.tar"
    echo ">> multi-platform manifest -> chameleon-ae-multiarch.tar"
fi

echo ">> buildx: $IMAGE:$TAG for $PLATFORMS"
docker buildx build \
    --platform "$PLATFORMS" \
    -t "$IMAGE:$TAG" \
    -f docker/Dockerfile \
    $OUT .
echo ">> done."
