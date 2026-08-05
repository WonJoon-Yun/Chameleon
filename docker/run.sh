#!/usr/bin/env bash
# Run a target inside the Chameleon artifact image, with the generated data and
# tables mounted back onto the host.
#
#   bash docker/run.sh                       # kick the tires (~2 min)
#   bash docker/run.sh make data             # CSV + XLSX export
#   bash docker/run.sh make all              # tables + macros + data export
#   bash docker/run.sh bash                  # interactive shell
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-chameleon-ae}"
TAG="${TAG:-latest}"

if ! docker image inspect "$IMAGE:$TAG" >/dev/null 2>&1; then
    echo ">> image $IMAGE:$TAG not found; building it first"
    bash docker/build.sh
fi

mkdir -p output

# -it only when there is a terminal on both ends: forcing it breaks every
# non-interactive use (a pipe, a CI job, `bash docker/run.sh make data > log`)
# with "cannot attach stdin to a TTY-enabled container".
TTY=()
[ -t 0 ] && [ -t 1 ] && TTY=(-it)

# Under ROOTFUL docker the container runs as root and would leave root-owned
# files in the bind-mounted output/, so run as the invoking user. Under ROOTLESS
# docker the daemon already maps container-root to the invoking user -- and
# passing a host uid outside the subuid range fails outright with
# "setgroups: invalid argument" -- so pass nothing there.
USERFLAG=(-u "$(id -u):$(id -g)")
if docker info -f '{{.SecurityOptions}}' 2>/dev/null | grep -q rootless; then
    USERFLAG=()
fi

# "${@:-make smoke}" would pass the default as ONE argv entry, so docker looked
# for an executable literally named "make smoke". Build the default as a proper
# argument list instead.
if [ "$#" -eq 0 ]; then
    set -- make smoke
fi

# macOS ships bash 3.2, where "${arr[@]}" on an EMPTY array is an "unbound
# variable" error under `set -u` (bash only made the empty case safe in 4.4).
# Both arrays above are empty in the ordinary case -- TTY whenever there is no
# terminal, USERFLAG under rootless docker -- so the plain expansion aborted the
# script before it reached docker, breaking exactly the non-interactive uses the
# TTY probe exists to support. ${arr[@]+"${arr[@]}"} expands to nothing at all
# when the array is empty and is safe on every bash.
exec docker run --rm ${TTY[@]+"${TTY[@]}"} ${USERFLAG[@]+"${USERFLAG[@]}"} \
    -v "$PWD/output:/artifact/output" \
    "$IMAGE:$TAG" \
    "$@"
