FROM pytorch/manylinux2_28-builder:cuda12.1 AS builder_amd64
FROM pytorch/manylinuxaarch64-builder:cuda12.4 AS builder_arm64

FROM builder_${TARGETARCH}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN set -eux; \
    yum install -y python3-pip; \
    pip3 install uv

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
