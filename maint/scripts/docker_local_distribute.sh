set -eux

# Get the CUDA version from the command line
IMAGE="tilelang-builder:manylinux"
docker build . -f "$(dirname "${BASH_SOURCE[0]}")/pypi.manylinux.Dockerfile" --tag ${IMAGE}

script="sh maint/scripts/local_distribution.sh"

docker run --rm -v $(pwd):/tilelang ${IMAGE} /bin/bash -c "$script"
