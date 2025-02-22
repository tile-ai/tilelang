#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Get the CUDA version from the command line
CUDA_VERSION=$1
if [ -z "${CUDA_VERSION}" ]; then
    echo "No CUDA version provided, using default 124"
    CUDA_VERSION=124
fi

IMAGE=pytorch/manylinux-cuda${CUDA_VERSION}

docker pull ${IMAGE}

# Run the docker container with the command directly
docker run --rm -v $(pwd):/tilelang ${IMAGE} /bin/bash -c "cd /tilelang && export PATH=/usr/local/bin:$PATH && yum install -y python3.8 python3.9 python3.10 python3.11 python3.12 && pip install -r requirements-build.txt && tox -e py38-pypi,py39-pypi,py310-pypi,py311-pypi,py312-pypi"
