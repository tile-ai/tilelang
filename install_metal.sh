rm -rf build

mkdir build

pushd build

CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache \
    cmake ..

make -j