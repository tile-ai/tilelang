# Get the CUDA version from the command line
IMAGE="tilelang-builder:manylinux"
docker build . -f "$(dirname "${BASH_SOURCE[0]}")/pypi.manylinux.Dockerfile" --tag ${IMAGE}

install_pip="python3.8 -m pip install --upgrade pip && python3.8 -m pip install -r requirements-build.txt"

tox_command="python3.8 -m tox -e py38-pypi,py39-pypi,py310-pypi,py311-pypi,py312-pypi"

docker run --rm --gpus all -v $(pwd):/tilelang ${IMAGE} /bin/bash -c "$install_pip && $tox_command"
