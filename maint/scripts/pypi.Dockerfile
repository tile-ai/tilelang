FROM nvidia/cuda:12.1.0-devel-ubuntu18.04

RUN set -eux; \
    apt-get update; \
    # Install gcc-9 and g++-9
    apt-get install -y software-properties-common; \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y; \
    apt-get update; \
    apt-get install -y wget curl libtinfo-dev zlib1g-dev libssl-dev build-essential \
                       libedit-dev libxml2-dev git gcc-9 g++-9; \
    # Switch default gcc/g++ to new version
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100; \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100; \
    update-alternatives --install /usr/bin/cc cc /usr/bin/gcc 100; \
    update-alternatives --install /usr/bin/c++ c++ /usr/bin/g++ 100; \
    gcc --version; g++ --version; \
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3; \
    rm Miniconda3-latest-Linux-x86_64.sh;

RUN apt-get update && apt-get install -y ninja-build

ENV PATH=/miniconda3/bin/:$PATH

# ✅ Accept Anaconda Terms of Service for both required channels
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create environments
RUN set -eux; \
    conda create -n py38 python=3.8 -y; \
    conda create -n py39 python=3.9 -y; \
    conda create -n py310 python=3.10 -y; \
    conda create -n py311 python=3.11 -y; \
    conda create -n py312 python=3.12 -y; \
    ln -s /miniconda3/envs/py38/bin/python3.8 /usr/bin/python3.8; \
    ln -s /miniconda3/envs/py39/bin/python3.9 /usr/bin/python3.9; \
    ln -s /miniconda3/envs/py310/bin/python3.10 /usr/bin/python3.10; \
    ln -s /miniconda3/envs/py311/bin/python3.11 /usr/bin/python3.11; \
    ln -s /miniconda3/envs/py312/bin/python3.12 /usr/bin/python3.12; \
    conda install -y cmake patchelf

WORKDIR /tilelang
