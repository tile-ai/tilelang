FROM pytorch/manylinux-builder:cuda12.1

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN set -eux; \
    yum -y update && yum install -y \
        zlib-devel openssl-devel \
        libedit-devel libxml2-devel \
        bzip2 bzip2-devel xz xz-devel \
        epel-release

RUN set -eux; \
    conda create -n py38 python=3.8 -y && \
    conda create -n py39 python=3.9 -y && \
    conda create -n py310 python=3.10 -y && \
    conda create -n py311 python=3.11 -y && \
    conda create -n py312 python=3.12 -y && \
    ln -sf /opt/conda/envs/py38/bin/python3.8 /usr/bin/python3.8 && \
    ln -sf /opt/conda/envs/py39/bin/python3.9 /usr/bin/python3.9 && \
    ln -sf /opt/conda/envs/py310/bin/python3.10 /usr/bin/python3.10 && \
    ln -sf /opt/conda/envs/py311/bin/python3.11 /usr/bin/python3.11 && \
    ln -sf /opt/conda/envs/py312/bin/python3.12 /usr/bin/python3.12 && \
    conda install -y cmake patchelf

WORKDIR /tilelang
