set -eux

# Get the CUDA version from the command line
IMAGE="tilelang-builder:manylinux"

# Detect host arch and map to Docker TARGETARCH values
HOST_UNAME=$(uname -m)
case "$HOST_UNAME" in
    x86_64)
        TARGETARCH=amd64
        ;;
    aarch64|arm64)
        TARGETARCH=arm64
        ;;
    *)
        echo "Unsupported architecture: $HOST_UNAME" >&2
        exit 1
        ;;
esac

# Prefer buildx for cross-arch builds if available
if docker buildx version >/dev/null 2>&1; then
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

    script="sh maint/scripts/local_distribution.sh"
    docker run --rm -v $(pwd):/tilelang "${TAG_IMAGE}" /bin/bash -c "$script"
    if [ -d dist ]; then
      mv -f dist "dist-${ARCH}"
    fi
  done
else
  echo "docker buildx not found; building only host arch: ${TARGETARCH}" >&2
  TAG_IMAGE="${IMAGE}-${TARGETARCH}"
  docker build \
    --build-arg TARGETARCH="$TARGETARCH" \
    . -f "$(dirname "${BASH_SOURCE[0]}")/pypi.manylinux.Dockerfile" --tag "${TAG_IMAGE}"

  script="sh maint/scripts/local_distribution.sh"
  docker run --rm -v $(pwd):/tilelang "${TAG_IMAGE}" /bin/bash -c "$script"
  if [ -d dist ]; then
    mv -f dist "dist-${TARGETARCH}"
  fi
fi
