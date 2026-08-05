# Containerised artifact

The image is self-contained: dependencies, source, shipped result records, and
calibration snapshots. Nothing is downloaded at run time.

## Supported architectures

| Platform | Docker platform | Status |
|---|---|---|
| ARM64 servers (AWS Graviton, Ampere) | `linux/arm64` | **built and tested** |
| Apple silicon M1–M4 (Docker Desktop) | `linux/arm64` | **built and tested** (native, same image) |
| Intel / AMD x86-64 (Linux, Windows WSL2) | `linux/amd64` | **built and tested under emulation** |
| Intel Mac | `linux/amd64` | same image as above; not run on native hardware |

Both images are built and `make smoke` passes on both, all four steps including
the golden-hash circuit regression and the live decoder cell. Every dependency
(`stim`, `pymatching`, `ldpc`, `chromobius`, `numpy`, `pandas`) publishes both
`manylinux_x86_64` and `manylinux_aarch64` wheels, so the same Dockerfile builds
on either architecture with no cross-compiler; the Dockerfile installs
`build-essential` so a source build succeeds where no wheel exists.

`make all` produces **byte-identical** CSV output on the two architectures (all
17 tables, matching MD5), so the exported numbers do not depend on the host
architecture.

One caveat on the x86-64 result: it was verified with `linux/amd64` under QEMU
emulation on an Apple silicon host, not on native x86-64 hardware. Emulation
exercises the same x86-64 wheels and the same code path, but it is not a
substitute for a native run, and it says nothing about native x86-64 timings.

## Build

```bash
bash docker/build.sh                  # native build for the current machine
bash docker/build.sh --multi          # linux/amd64 + linux/arm64 (OCI tarball)
IMAGE=ghcr.io/<user>/chameleon-ae bash docker/build.sh --multi --push
```

On Apple silicon use the plain native build. Docker Desktop resolves
`linux/arm64` automatically; `--platform linux/amd64` would force QEMU emulation
and slow the decoders down by roughly an order of magnitude.

`--multi` builds for a platform the host cannot execute natively, so it needs
binfmt/QEMU registered or a remote builder for the other architecture. Check
what the daemon can reach with `docker buildx ls`: if it lists only the native
platform, register the handlers with

```bash
docker run --privileged --rm tonistiigi/binfmt --install all
```

Docker Desktop (macOS and Windows) ships these handlers already, so `linux/amd64`
is listed out of the box on Apple silicon and the cross build works with no extra
setup — that is how the x86-64 image above was built and tested. A bare rootless
daemon on an aarch64 host generally has no `qemu-x86_64` handler and needs the
`binfmt` step first.

## Run

```bash
bash docker/run.sh                    # kick the tires (~2 min)
bash docker/run.sh make data          # CSV + XLSX export
bash docker/run.sh make all           # tables + macros + data export
bash docker/run.sh bash               # interactive shell
```

`docker/run.sh` bind-mounts `./output`, so the CSV files and XLSX workbook appear
on the host with the invoking user's ownership.

Equivalent Compose targets:

```bash
docker compose -f docker/docker-compose.yml run --rm smoke
docker compose -f docker/docker-compose.yml run --rm all
```

## Resources

| Stage | Cores | RAM | Wall time |
|---|---|---|---|
| `make smoke` | 1 | < 500 MB | ~2 min |
| `make all` | 1 | < 250 MB | ~15 s |
| `make experiments` (full re-measurement) | 32–160 | 32 GB+ | core-days |

Give the container more cores for the full re-measurement:

```bash
docker run --rm -it --cpus=32 -v "$PWD/output:/artifact/output" \
    chameleon-ae make experiments PROCS=32
```
