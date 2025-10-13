# Installation Guide

## Installing with pip

**Prerequisites for installation via wheel or PyPI:**

- **Operating System**: Ubuntu 20.04 or later
- **Python Version**: >= 3.8
- **CUDA Version**: >= 11.0

The easiest way to install **tile-lang** is directly from PyPI using pip. To install the latest version, run the following command in your terminal:

```bash
pip install tilelang
```

Alternatively, you may choose to install **tile-lang** using prebuilt packages available on the Release Page:

```bash
pip install tilelang-0.0.0.dev0+ubuntu.20.4.cu120-py3-none-any.whl
```

To install the latest version of **tile-lang** from the GitHub repository, you can run the following command:

```bash
pip install git+https://github.com/tile-ai/tilelang.git
```

After installing **tile-lang**, you can verify the installation by running:

```bash
python -c "import tilelang; print(tilelang.__version__)"
```

## Building from Source

**Prerequisites for building from source:**

- **Operating System**: Linux
- **Python Version**: >= 3.7
- **CUDA Version**: >= 10.0
- **LLVM**: < 20 if you are using the bundled TVM submodule

We recommend using a Docker container with the necessary dependencies to build **tile-lang** from source. You can use the following command to run a Docker container with the required dependencies:

```bash
docker run --gpus all -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.01-py3
```

To build and install **tile-lang** directly from source, follow these steps. This process requires certain pre-requisites from Apache TVM, which can be installed on Ubuntu/Debian-based systems using the following commands:

```bash
apt-get update
apt-get install -y python3 python3-dev python3-setuptools gcc zlib1g-dev build-essential cmake libedit-dev
```

After installing the prerequisites, you can clone the **tile-lang** repository and install it using pip:

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
pip install .  # Please be patient, this may take some time.
```

If you want to install **tile-lang** in development mode, you can run the following command:

```bash
pip install -e .
```

We currently provide four methods to install **tile-lang**:

1. [Install Using Docker](#install-method-1) (Recommended)
2. [Install from Source (using your own TVM installation)](#install-method-2)
3. [Install from Source (using the bundled TVM submodule)](#install-method-3)
4. [Install Using the Provided Script](#install-method-4)

(install-method-1)=

### Method 1: Install Using Docker (Recommended)

For users who prefer a containerized environment with all dependencies pre-configured, **tile-lang** provides Docker images for different CUDA versions. This method is particularly useful for ensuring consistent environments across different systems and is the **recommended approach** for most users.

**Prerequisites:**
- Docker installed on your system
- NVIDIA Docker runtime (nvidia-docker2) for GPU support
- Compatible NVIDIA GPU (e.g., B200, H100, etc.)

1. **Clone the Repository**:

```bash
git clone --recursive https://github.com/tile-ai/tilelang
cd tilelang
```

2. **Build Docker Image**:

Navigate to the docker directory and build the image for your desired CUDA version:

```bash
cd docker
docker build -f Dockerfile.cu120 -t tilelang-cu120 .
```

Available Dockerfiles:
- `Dockerfile.cu120` - For CUDA 12.0
- Other CUDA versions may be available in the docker directory

3. **Run Docker Container**:

Start the container with GPU access and volume mounting:

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /home/tilelang:/home/tilelang \
  --name tilelang_b200 \
  tilelang-cu120 \
  /bin/zsh
```

**Command Parameters Explanation:**
- `--shm-size 32g`: Increases shared memory size for better performance
- `--gpus all`: Enables access to all available GPUs
- `-v /home/tilelang:/home/tilelang`: Mounts host directory to container (adjust path as needed)
- `--name tilelang_b200`: Assigns a name to the container for easy management
- `/bin/zsh`: Uses zsh as the default shell

4. **Access the Container**:

```bash
docker exec -it tilelang_b200 /bin/zsh
```

5. **Verify Installation**:

Once inside the container, verify that **tile-lang** is working correctly:

```bash
python -c "import tilelang; print(tilelang.__version__)"
```

You can now run TileLang examples and develop your applications within the containerized environment. The Docker image comes with all necessary dependencies pre-installed, including CUDA toolkit, TVM, and TileLang itself.

**Example Usage:**

After accessing the container, you can run TileLang examples:

```bash
cd /home/tilelang/examples
python elementwise/test_example_elementwise.py
```

This Docker-based installation method provides a complete, isolated environment that works seamlessly on systems with compatible NVIDIA GPUs like the B200, ensuring optimal performance for TileLang applications.

(install-method-2)=

### Method 2: Install from Source (Using Your Own TVM Installation)

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

(install-method-3)=

### Method 3: Install from Source (Using the Bundled TVM Submodule)

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
# echo "set(USE_LLVM ON)"  # set USE_LLVM to ON if using LLVM
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

(install-method-4)=

### Method 4: Install Using the Provided Script

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
```

## Install with Nightly Version

For users who want access to the latest features and improvements before official releases, we provide nightly builds of **tile-lang**.

```bash
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu121/
# or pip install tilelang --find-links https://tile-ai.github.io/whl/nightly/cu121/
```

> **Note:** Nightly builds contain the most recent code changes but may be less stable than official releases. They're ideal for testing new features or if you need a specific bugfix that hasn't been released yet.
