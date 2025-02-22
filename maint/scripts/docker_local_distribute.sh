#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Get the CUDA version from the command line
IMAGE=nvidia/cuda:12.1.0-devel-ubuntu18.04

docker pull ${IMAGE}

# Run the docker container with the command directly
# docker run --rm -v $(pwd):/tilelang ${IMAGE} /bin/bash -c "cd /tilelang && export PATH=/usr/local/bin:$PATH export CC=/opt/rh/devtoolset-9/root/usr/bin/gcc && export CXX=/opt/rh/devtoolset-9/root/usr/bin/gcc && expprt CUDAHOSTCXX=/usr/bin/g++ && /opt/python/cp38-cp38/bin/python -m pip install -r requirements-build.txt && /opt/python/cp38-cp38/bin/python -m tox -e py38,py39,py310,py311,py312"


# docker run --rm -v $(pwd):/tilelang ${IMAGE} /bin/bash -it

command="apt update && apt install -y software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt install -y wget python3.8 python3.8-dev python3-pip python3-setuptools libtinfo-dev zlib1g-dev libssl-dev build-essential libedit-dev libxml2-dev && ln -s /usr/bin/python3.8 /usr/bin/python && wget https://github.com/Kitware/CMake/releases/download/v3.28.4/cmake-3.28.4-linux-x86_64.tar.gz && tar -xvzf cmake-*.tar.gz && rm cmake-*.tar.gz && cd cmake-* && cp bin/* /usr/local/bin/ && mv share/* /usr/local/share/ && mv man/* /usr/local/man/ && hash -r && cd /tilelang && export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH && python3.8 -m pip install  --upgrade pip && python3.8 -m pip install -r requirements-build.txt && python3.8 -m tox -e py38,py39,py310,py311,py312" 


docker run --rm -v $(pwd):/tilelang ${IMAGE} /bin/bash -c "$command"
