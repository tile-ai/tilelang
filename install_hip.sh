#!/bin/bash

# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

echo "Starting installation script..."

# check if ROCm environment exists
if [ ! -d "/opt/rocm" ]; then
    echo "Error: ROCm installation not found at /opt/rocm."
    echo "Please install ROCm first. Visit https://rocm.docs.amd.com/en/latest/deploy/linux/installer/install.html for instructions."
    exit 1
fi

# set HIP_HOME environment variable
export HIP_HOME="/opt/rocm"
echo "HIP_HOME is set to $HIP_HOME"
pip install -r requirements-build.txt
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install Python requirements."
    exit 1
else
    echo "Python requirements installed successfully."
fi
# determine if root
USER_IS_ROOT=false
if [ "$EUID" -eq 0 ]; then
    USER_IS_ROOT=true
fi

if $USER_IS_ROOT; then
    # Fetch the GPG key for the LLVM repository and add it to the trusted keys
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc

    # Check if the repository is already present in the sources.list
    if ! grep -q "http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" /etc/apt/sources.list; then
        # Add the LLVM repository to sources.list
        echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list
        echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" >> /etc/apt/sources.list
    else
        # Print a message if the repository is already added
        echo "The repository is already added."
    fi

    # Update package lists and install llvm-16
    apt-get update
    apt-get install -y llvm-16
else
    # Fetch the GPG key for the LLVM repository and add it to the trusted keys using sudo
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | sudo tee /etc/apt/trusted.gpg.d/apt.llvm.org.asc

    # Check if the repository is already present in the sources.list
    if ! grep -q "http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" /etc/apt/sources.list; then
        # Add the LLVM repository to sources.list using sudo
        echo "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" | sudo tee -a /etc/apt/sources.list
        echo "deb-src http://apt.llvm.org/focal/ llvm-toolchain-focal-16 main" | sudo tee -a /etc/apt/sources.list
    else
        # Print a message if the repository is already added
        echo "The repository is already added."
    fi

    # Update package lists and install llvm-16 using sudo
    sudo apt-get update
    sudo apt-get install -y llvm-16
fi

# Step 9: Clone and build TVM
echo "Cloning TVM repository and initializing submodules..."
# clone and build tvm
git submodule update --init --recursive

if [ -d build ]; then
    rm -rf build
fi

mkdir build
cp 3rdparty/tvm/cmake/config.cmake build
cd build


echo "Configuring TVM build with LLVM and HIP..."
echo "set(USE_LLVM llvm-config-16)" >> config.cmake
echo "set(USE_ROCM ${HIP_HOME})" >> config.cmake
echo "set(USE_CUDA OFF)" >> config.cmake


echo "Running CMake for TileLang..."
cmake .. -DUSE_HIP=ON
if [ $? -ne 0 ]; then
    echo "Error: CMake configuration failed."
    exit 1
fi

echo "Building TileLang with make..."
make -j
if [ $? -ne 0 ]; then
    echo "Error: TileLang build failed."
    exit 1
else
    echo "TileLang build completed successfully."
fi

cd ..


# Define the lines to be added
TILELANG_PATH="$(pwd)"
echo "Configuring environment variables for TVM..."
echo "export PYTHONPATH=${TILELANG_PATH}:\$PYTHONPATH" >> ~/.bashrc
TVM_HOME_ENV="export TVM_HOME=${TILELANG_PATH}/3rdparty/tvm"
TILELANG_PYPATH_ENV="export PYTHONPATH=\$TVM_HOME/python:${TILELANG_PATH}:\$PYTHONPATH"
HIP_HOME_ENV="export HIP_HOME=${HIP_HOME}"
USE_HIP_ENV="export USE_HIP=True"

# Check and add the first line if not already present
if ! grep -qxF "$TVM_HOME_ENV" ~/.bashrc; then
    echo "$TVM_HOME_ENV" >> ~/.bashrc
    echo "Added TVM_HOME to ~/.bashrc"
else
    echo "TVM_HOME is already set in ~/.bashrc"
fi

# Check and add the second line if not already present
if ! grep -qxF "$TILELANG_PYPATH_ENV" ~/.bashrc; then
    echo "$TILELANG_PYPATH_ENV" >> ~/.bashrc
    echo "Added PYTHONPATH to ~/.bashrc"
else
    echo "PYTHONPATH is already set in ~/.bashrc"
fi

# Check and add the third line if not already present
if ! grep -qxF "$HIP_HOME_ENV" ~/.bashrc; then
    echo "$HIP_HOME_ENV" >> ~/.bashrc
    echo "Added HIP_HOME to ~/.bashrc"
else
    echo "HIP_HOME is already set in ~/.bashrc"
fi

if ! grep -qxF "$USE_HIP_ENV" ~/.bashrc; then
    echo "$USE_HIP_ENV" >> ~/.bashrc
    echo "Added USE_HIP to ~/.bashrc"
else
    echo "USE_HIP is already set in ~/.bashrc"
fi

# Reload ~/.bashrc to apply the changes
source ~/.bashrc

echo "Installation script completed successfully."

echo "Installing TileLang package with HIP support..."
pip install -e . -v USE_HIP=True

echo "HIP installation script completed successfully."
echo "TileLang is now configured to use HIP for AMD GPU support."
