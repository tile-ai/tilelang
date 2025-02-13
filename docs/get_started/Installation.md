# Installation Guide

## Installing with pip

**Prerequisites for installation via wheel or PyPI:**

- **Operating System**: Ubuntu 20.04 or later
- **Python Version**: >= 3.8
- **CUDA Version**: >= 11.0

The easiest way to install TileLang is directly from PyPI using pip. To install the latest version, run the following command in your terminal:

```bash
pip install tilelang
```

Alternatively, you may choose to install TileLang using prebuilt packages available on the Release Page:

```bash
pip install tilelang-0.0.0.dev0+ubuntu.20.4.cu120-py3-none-any.whl
```

To install the latest version of TileLang from the GitHub repository, you can run the following command:

```bash
pip install git+https://github.com/tile-ai/tilelang.git
```

After installing TileLang, you can verify the installation by running:

```bash
python -c "import tilelang; print(tilelang.__version__)"
```

## Building from Source

**Prerequisites for building from source:**

- **Operating System**: Linux
- **Python Version**: >= 3.7
- **CUDA Version**: >= 10.0

We recommend using a Docker container with the necessary dependencies to build TileLang from source. You can use the following command to run a Docker container with the required dependencies:

```bash
docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.01-py3
```

To build and install TileLang directly from source, follow these steps. This process requires certain pre-requisites from Apache TVM, which can be installed on Ubuntu/Debian-based systems using the following commands:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

After installing the prerequisites, you can clone the TileLang repository and install it using pip:

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tileLang
pip install .  # Please be patient, this may take some time.
```

If you want to install TileLang in development mode, you can run the following command:

```bash
pip install -e .
```

We currently provide three methods to install **TileLang**:

1. [Install from Source (using your own TVM installation)](#install-method-1)
2. [Install from Source (using the bundled TVM submodule)](#install-method-2)
3. [Install Using the Provided Script](#install-method-3)

(install-method-1)=

### Method 1: Install from Source (Using Your Own TVM Installation)

If you already have a compatible TVM installation, follow these steps:

1. **Clone the Repository**:

```bash
git clone --recursive https://github.com/tile-ai/tilelang
cd tilelang
```

**Note**: Use the `--recursive` flag to include necessary submodules.

2. **Configure Build Options**:

Create a build directory and specify your existing TVM path:

```bash
mkdir build
cd build
cmake .. -DTVM_PREBUILD_PATH=/your/path/to/tvm/build  # e.g., /workspace/tvm/build
make -j 16
```

3. **Set Environment Variables**:

Update `PYTHONPATH` to include the `tile-lang` Python module:

```bash
export PYTHONPATH=/your/path/to/tilelang/:$PYTHONPATH
# TVM_IMPORT_PYTHON_PATH is used by 3rd-party frameworks to import TVM
export TVM_IMPORT_PYTHON_PATH=/your/path/to/tvm/python
```

(install-method-2)=

### Method 2: Install from Source (Using the Bundled TVM Submodule)

If you prefer to use the built-in TVM version, follow these instructions:

1. **Clone the Repository**:

```bash
git clone --recursive https://github.com/tile-ai/tilelang
cd tilelang
```

**Note**: Ensure the `--recursive` flag is included to fetch submodules.

2. **Configure Build Options**:

Copy the configuration file and enable the desired backends (e.g., LLVM and CUDA):

```bash
mkdir build
cp 3rdparty/tvm/cmake/config.cmake build
cd build
echo "set(USE_LLVM ON)" >> config.cmake
echo "set(USE_CUDA ON)" >> config.cmake 
# or echo "set(USE_ROCM ON)" >> config.cmake to enable ROCm runtime
cmake ..
make -j 16
```

The build outputs (e.g., `libtilelang.so`, `libtvm.so`, `libtvm_runtime.so`) will be generated in the `build` directory.

3. **Set Environment Variables**:

Ensure the `tile-lang` Python package is in your `PYTHONPATH`:

```bash
export PYTHONPATH=/your/path/to/tilelang/:$PYTHONPATH
```

(install-method-3)=

### Method 3: Install Using the Provided Script

For a simplified installation, use the provided script:

1. **Clone the Repository**:

```bash
git clone --recursive https://github.com/tile-ai/tilelang
cd tilelang
```

2. **Run the Installation Script**:

```bash
bash install_cuda.sh
# or bash `install_amd.sh` if you want to enable ROCm runtime
