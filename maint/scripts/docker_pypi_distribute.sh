#!/usr/bin/env bash
set -euxo pipefail

IMAGE="tilelang-builder:manylinux"

HOST_UNAME=$(uname -m)
case "$HOST_UNAME" in
  x86_64) TARGETARCH=amd64 ;;
  aarch64|arm64) TARGETARCH=arm64 ;;
  *) echo "Unsupported architecture: $HOST_UNAME" >&2; exit 1 ;;
esac

if docker buildx version >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    docker run --rm --privileged tonistiigi/binfmt --install amd64,arm64 >/dev/null 2>&1 || true
  fi

  if ! docker buildx inspect multi >/dev/null 2>&1; then
    docker buildx create --name multi --driver docker-container --use >/dev/null 2>&1 || true
  else
    docker buildx use multi >/dev/null 2>&1 || true
  fi
  docker buildx inspect --bootstrap >/dev/null 2>&1 || true

  for ARCH in amd64 arm64; do
    TAG_PLATFORM="linux/${ARCH}"
    TAG_IMAGE="${IMAGE}-${ARCH}"

    docker buildx build \
      --platform "${TAG_PLATFORM}" \
      --build-arg TARGETARCH="${ARCH}" \
      -f "$(dirname "${BASH_SOURCE[0]}")/pypi.manylinux.Dockerfile" \
      -t "${TAG_IMAGE}" \
      --load \
      .

    script="sh maint/scripts/pypi_distribution.sh"
    docker run --rm \
      --platform "${TAG_PLATFORM}" \
      -v "$(pwd):/tilelang" \
      "${TAG_IMAGE}" \
      /bin/bash -lc "$script"

    if [ -d dist ]; then
      mv -f dist "dist-pypi-${ARCH}"
    fi
  done

else
  echo "docker buildx not found; building only host arch: ${TARGETARCH}" >&2
  TAG_IMAGE="${IMAGE}-${TARGETARCH}"
  TAG_PLATFORM="linux/${TARGETARCH}"

  docker build \
    --build-arg TARGETARCH="$TARGETARCH" \
    -f "$(dirname "${BASH_SOURCE[0]}")/pypi.manylinux.Dockerfile" \
    -t "${TAG_IMAGE}" \
    .

  script="sh maint/scripts/pypi_distribution.sh"
  docker run --rm \
    --platform "${TAG_PLATFORM}" \
    -v "$(pwd):/tilelang" \
    "${TAG_IMAGE}" \
    /bin/bash -lc "$script"

  if [ -d dist ]; then
    mv -f dist "dist-pypi-${TARGETARCH}"
  fi
fi
