#!/bin/bash

# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# install requirements
pip install -r requirements.txt

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


echo "Configuring TVM build with LLVM and CUDA paths..."
echo "set(USE_LLVM $LLVM_CONFIG_PATH)" >> config.cmake && echo "set(USE_ROCM /opt/rocm)" >> config.cmake

echo "Running CMake for TileLang..."
cmake ..
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
echo "export CUDA_DEVICE_ORDER=PCI_BUS_ID" >> ~/.bashrc
TVM_HOME_ENV="export TVM_HOME=${TILELANG_PATH}/3rdparty/tvm"
TILELANG_PYPATH_ENV="export PYTHONPATH=\$TVM_HOME/python:${TILELANG_PATH}:\$PYTHONPATH"
CUDA_DEVICE_ORDER_ENV="export CUDA_DEVICE_ORDER=PCI_BUS_ID"

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
if ! grep -qxF "$CUDA_DEVICE_ORDER_ENV" ~/.bashrc; then
    echo "$CUDA_DEVICE_ORDER_ENV" >> ~/.bashrc
    echo "Added CUDA_DEVICE_ORDER to ~/.bashrc"
else
    echo "CUDA_DEVICE_ORDER is already set in ~/.bashrc"
fi

# Reload ~/.bashrc to apply the changes
source ~/.bashrc

echo "Installation script completed successfully."
