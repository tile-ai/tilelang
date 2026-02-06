#!/bin/bash

if [ -z "${NVSHMEM_SRC}" ]; then
    export NVSHMEM_SRC="$(realpath ../../3rdparty/nvshmem_src)"
    echo "NVSHMEM_SRC not set, defaulting to ${NVSHMEM_SRC}"
else
    NVSHMEM_SRC="$(realpath ${NVSHMEM_SRC})"
    echo "Using NVSHMEM_SRC=${NVSHMEM_SRC}"
fi

if [ -d "${NVSHMEM_SRC}" ]; then
    if [ "$(ls -A ${NVSHMEM_SRC})" ]; then
        echo "NVSHMEM_SRC directory (${NVSHMEM_SRC}) is not empty, cleaning it..."
        rm -rf "${NVSHMEM_SRC}/"*
        rm -rf "${NVSHMEM_SRC}/".* 2>/dev/null || true
    fi
else
    mkdir -p "${NVSHMEM_SRC}"
fi

wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
tar zxvf nvshmem_src_3.2.5-1.txz
rm -rf nvshmem_src_3.2.5-1.txz

mkdir -p "${NVSHMEM_SRC}"
mv nvshmem_src/* "${NVSHMEM_SRC}/"
mv nvshmem_src/.* "${NVSHMEM_SRC}/" 2>/dev/null || true
rmdir nvshmem_src


export NVSHMEM_PATH="${NVSHMEM_SRC}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}")
echo "SCRIPT_DIR: ${SCRIPT_DIR}"
echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "NVSHMEM will be installed to: ${NVSHMEM_SRC}"

ARCH=""
JOBS=""

# Iterate over the command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --arch)
        # Process the arch argument
        ARCH="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    --jobs)
        # Process the jobs argument
        JOBS="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    *)
        # Unknown argument
        echo "Unknown argument: $1"
        shift # Skip the argument
        ;;
    esac
done

if [[ -n "${ARCH}" ]]; then
    export CMAKE_CUDA_ARCHITECTURES="${ARCH}"
    CUDAARCH_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${ARCH}"
fi

if [[ -z "${JOBS}" ]]; then
    JOBS=$(nproc --ignore 2)
fi

export NVSHMEM_IBGDA_SUPPORT=0
export NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=0
export NVSHMEM_IBDEVX_SUPPORT=0
export NVSHMEM_IBRC_SUPPORT=1
export NVSHMEM_LIBFABRIC_SUPPORT=0
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_TORCH_SUPPORT=1
export NVSHMEM_ENABLE_ALL_DEVICE_INLINING=1

pushd "${NVSHMEM_SRC}"
mkdir -p build
cd build
CMAKE=${CMAKE:-cmake}

if [ ! -f CMakeCache.txt ]; then
    ${CMAKE} .. \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        ${CUDAARCH_ARGS} \
        -DNVSHMEM_BUILD_TESTS=OFF \
        -DNVSHMEM_BUILD_EXAMPLES=OFF \
        -DNVSHMEM_BUILD_PACKAGES=OFF
fi

make VERBOSE=1 -j"${JOBS}"
popd

echo "NVSHMEM installed successfully to ${NVSHMEM_SRC}"
