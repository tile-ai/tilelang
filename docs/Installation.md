# Installation Guide

## Installing with pip

**Prerequisites for installation via wheel or PyPI:**
- **Operating System**: Ubuntu 20.04 or later
- **Python Version**: >= 3.8
- **CUDA Version**: >= 11.0

The easiest way to install TileLang is directly from the PyPi using pip. To install the latest version, run the following command in your terminal.

**Note**: Currently, TileLang whl is only supported on Ubuntu 20.04 or later version as we build the whl files on this platform. Currently we only provide whl files for CUDA>=11.0 and with Python>=3.8. **If you are using a different platform or environment, you may need to [build TileLang from source](https://github.com/tile-ai/tilelang/blob/main/docs/Installation.md#building-from-source).**

```bash
pip install tilelang
```

Alternatively, you may choose to install TileLang using prebuilt packages available on the Release Page:

```bash
pip install tilelang-0.0.0.dev0+ubuntu.20.4.cu120-py3-none-any.whl
```

To install the latest version of TileLang from the github repository, you can run the following command:

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

We recommend using a docker container with the necessary dependencies to build TileLang from source. You can use the following command to run a docker container with the necessary dependencies:

```bash
docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.01-py3
```

To build and install TileLang directly from source, follow the steps below. This process requires certain pre-requisites from apache tvm, which can be installed on Ubuntu/Debian-based systems using the following commands:

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

After installing the prerequisites, you can clone the TileLang repository and install it using pip:

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
cd TileLang
pip install .  # Please be patient, this may take some time.
```

if you want to install TileLang with the development mode, you can run the following command:

```bash
pip install -e .
```

We currently provide three ways to install **tile-lang**:
 - [Install from Source (using your own TVM installation)](#install-from-source-with-your-own-tvm-installation)
 - [Install from Source (using the bundled TVM submodule)](#install-from-source-with-our-tvm-submodule)
 - [Install Using the Provided Script](#install-with-provided-script)


### Method 1: Install from Source (using your own TVM installation)

If you already have a compatible TVM installation, follow these steps:

1. **Clone the Repository:**

    ```bash
    git clone --recursive https://github.com/tile-ai/tilelang
    cd TileLang
    ```

   > **Note**: Use the `--recursive` flag to include necessary submodules.

2. **Configure Build Options:**

    Create a build directory and specify your existing TVM path:

    ```bash
    mkdir build
    cd build
    cmake .. -DTVM_PREBUILD_PATH=/your/path/to/tvm/build  # e.g., /workspace/tvm/build
    make -j 16
    ```

3. **Set Environment Variables:**

    Update `PYTHONPATH` to include the `tile-lang` Python module:

    ```bash
    export PYTHONPATH=/your/path/to/tile-lang/python:$PYTHONPATH
    # TVM_IMPORT_PYTHON_PATH is used by 3rdparty framework to import tvm
    export TVM_IMPORT_PYTHON_PATH=/your/path/to/tvm/python
    ```

### Method 2: Install from Source (using the bundled TVM submodule)

If you prefer to use the built-in TVM version, follow these instructions:

1. **Clone the Repository:**

    ```bash
    git clone --recursive https://github.com/tile-ai/tilelang
    cd TileLang
    ```

   > **Note**: Ensure the `--recursive` flag is included to fetch submodules.

2. **Configure Build Options:**

    Copy the configuration file and enable the desired backends (e.g., LLVM and CUDA):

    ```bash
    mkdir build
    cp 3rdparty/tvm/cmake/config.cmake build
    cd build
    echo "set(USE_LLVM ON)" >> config.cmake
    echo "set(USE_CUDA ON)" >> config.cmake 
    # or echo "set(USE_ROCM ON)" >> config.cmake if want to enable rocm runtime
    cmake ..
    make -j 16
    ```

   The build outputs (e.g., `libtilelang.so`, `libtvm.so`, `libtvm_runtime.so`) will be generated in the `build` directory.

3. **Set Environment Variables:**

    Ensure the `tile-lang` Python package is in your `PYTHONPATH`:

    ```bash
    export PYTHONPATH=/your/path/to/TileLang/python:$PYTHONPATH
    ```

### Method 3: Install Using the Provided Script

For a simplified installation, use the provided script:

1. **Clone the Repository:**

    ```bash
    git clone --recursive https://github.com/tile-ai/tilelang
    cd TileLang
    ```

2. **Run the Installation Script:**

    ```bash
    bash install.sh
    # or bash `install_amd.sh` if you want to enable rocm runtime
    ```

This script automates the setup, including submodule initialization and configuration.
