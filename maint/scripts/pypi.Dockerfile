FROM nvidia/cuda:12.1.0-devel-ubuntu18.04

RUN set -eux; \
    apt-get update; \
    apt-get install -y wget curl libtinfo-dev zlib1g-dev libssl-dev build-essential libedit-dev libxml2-dev; \
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3; \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH=/miniconda3/bin/:$PATH

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