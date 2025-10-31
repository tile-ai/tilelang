ARG TARGETARCH
FROM pytorch/manylinux2_28-builder:cuda12.1 AS builder_amd64
ENV CUDA_VERSION=12.1 \
    AUDITWHEEL_PLAT=manylinux_2_28_x86_64
RUN pip3 install uv

FROM pytorch/manylinuxaarch64-builder:cuda12.8 AS builder_arm64
ENV CUDA_VERSION=12.8 \
    AUDITWHEEL_PLAT=manylinux_2_28_aarch64
RUN /opt/python/cp312-cp312/bin/pip install uv

FROM builder_${TARGETARCH}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

ENV PATH="/usr/local/cuda/bin:${PATH}"

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN set -eux; \
    uv venv -p 3.12 --seed /venv; \
    git config --global --add safe.directory '/tilelang'

ENV PATH="/venv/bin:$PATH" \
    VIRTUAL_ENV=/venv

RUN uv pip install build wheel

WORKDIR /tilelang
