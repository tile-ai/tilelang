FROM pytorch/manylinux2_28-builder:cuda12.1 AS builder_amd64
ENV CUDA_VERSION=12.1 \
    AUDITWHEEL_PLAT=manylinux_2_28_x86_64
FROM pytorch/manylinuxaarch64-builder:cuda12.4 AS builder_arm64
ENV CUDA_VERSION=12.4 \
    AUDITWHEEL_PLAT=manylinux_2_28_aarch64

FROM builder_${TARGETARCH}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

RUN set -eux; \
    pip3 install uv; \
    uv venv -p 3.8 --seed /venv; \
    git config --global --add safe.directory '/tilelang'

ENV PATH="/venv/bin:$PATH" \
    VIRTUAL_ENV=/venv

RUN uv pip install build wheel

WORKDIR /tilelang
