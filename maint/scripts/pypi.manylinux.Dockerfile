FROM quay.io/pypa/manylinux_2_28_x86_64 AS builder_amd64

RUN dnf config-manager --add-repo https://developer.download.nvidia.cn/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

ARG CUDA_VERSION=12.8
ENV CUDA_VERSION=${CUDA_VERSION}

FROM quay.io/pypa/manylinux_2_28_aarch64 AS builder_arm64

RUN dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo

ARG CUDA_VERSION=12.8
ENV CUDA_VERSION=${CUDA_VERSION}

ARG TARGETARCH
FROM builder_${TARGETARCH}

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC

ENV PATH="/usr/local/cuda/bin:${PATH}"

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

RUN set -eux; \
    pipx install cibuildwheel; \
    git config --global --add safe.directory '/tilelang'

WORKDIR /tilelang
