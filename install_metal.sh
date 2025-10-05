#!/bin/bash

set -eux

git submodule update --init --recursive

rm -rf build

mkdir build
cp 3rdparty/tvm/cmake/config.cmake build
cd build

echo "set(USE_METAL ON)" >> config.cmake

CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache cmake ..

CORES=$(sysctl -n hw.logicalcpu)
MAKE_JOBS=$(( CORES / 2 ))
make -j${MAKE_JOBS}
