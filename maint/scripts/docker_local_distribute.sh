#!/bin/bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

# Get the CUDA version from the command line
IMAGE="tilelang-builder:18.04"
docker build . -f "$(dirname "${BASH_SOURCE[0]}")/pypi.Dockerfile" --tag ${IMAGE}

install_pip="python3.8 -m pip install --upgrade pip && python3.8 -m pip install -r requirements-build.txt"

tox_command="python3.8 -m tox -e py38,py39,py310,py311,py312"

docker run --rm -v $(pwd):/tilelang ${IMAGE} /bin/bash -c "$install_pip && $tox_command"
