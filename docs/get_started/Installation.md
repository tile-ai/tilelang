# Installation Guide

## Installing with pip

**Prerequisites for installation via wheel or PyPI:**

- **glibc**: 2.28 (Ubuntu 20.04 or later)
- **Python Version**: >= 3.8
- **CUDA Version**: 12.0 <= CUDA < 13

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
- **Python Version**: >= 3.8
- **CUDA Version**: >= 10.0

```bash
docker run -it --rm --ipc=host nvcr.io/nvidia/pytorch:23.01-py3
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
pip install . -v
```

If you want to install **tile-lang** in development mode, you can run the following command:

```bash
pip install -e . -v
```

If you prefer to work directly from the source tree via `PYTHONPATH`, make sure the native extension is built first:

```bash
mkdir -p build
cd build
cmake .. -DUSE_CUDA=ON
make -j
```
Then add the repository root to `PYTHONPATH` before importing `tilelang`, for example:

```bash
export PYTHONPATH=/path/to/tilelang:$PYTHONPATH
python -c "import tilelang; print(tilelang.__version__)"
```

Some useful CMake options you can toggle while configuring:
- `-DUSE_CUDA=ON|OFF` builds against NVIDIA CUDA (default ON when CUDA headers are found).
- `-DUSE_ROCM=ON` selects ROCm support when building on AMD GPUs.
- `-DNO_VERSION_LABEL=ON` disables the backend/git suffix in `tilelang.__version__`.

We currently provide four methods to install **tile-lang**:

1. [Install Using Docker](#install-method-1) (Recommended)
2. [Install from Source (using the bundled TVM submodule)](#install-method-2)
3. [Install from Source (using your own TVM installation)](#install-method-3)

(install-method-1)=

### Method 1: Install Using Docker (Recommended)

For users who prefer a containerized environment with all dependencies pre-configured, **tile-lang** provides Docker images for different CUDA versions. This method is particularly useful for ensuring consistent environments across different systems and is the **recommended approach** for most users.

**Prerequisites:**
- Docker installed on your system
- NVIDIA Docker runtime or GPU is not necessary for building tilelang, you can build on a host without GPU and use that built image on other machine.

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

### Method 2: Install from Source (Using the Bundled TVM Submodule)

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
pip install . -v
```

(install-method-3)=

### Method 3: Install from Source (Using Your Own TVM Installation)

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
TVM_ROOT=<your-tvm-repo> pip install . -v
```

## Install with Nightly Version

For users who want access to the latest features and improvements before official releases, we provide nightly builds of **tile-lang**.

```bash
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu121/
# or pip install tilelang --find-links https://tile-ai.github.io/whl/nightly/cu121/
```

> **Note:** Nightly builds contain the most recent code changes but may be less stable than official releases. They're ideal for testing new features or if you need a specific bugfix that hasn't been released yet.

## Install Configs

### Build-time environment variables
`USE_CUDA`: If to enable CUDA support, default: `ON` on Linux, set to `OFF` to build a CPU version. By default, we'll use `/usr/local/cuda` for building tilelang. Set `CUDAToolkit_ROOT` to use different cuda toolkit.

`USE_ROCM`: If to enable ROCm support, default: `OFF`. If your ROCm SDK does not located in `/opt/rocm`, set `USE_ROCM=<rocm_sdk>` to enable build ROCm against custom sdk path.

`USE_METAL`: If to enable Metal support, default: `ON` on Darwin.

`TVM_ROOT`: TVM source root to use.

`NO_VERSION_LABEL` and `NO_TOOLCHAIN_VERSION`:
When building tilelang, we'll try to embed SDK and version information into package version as below,
where local version label could look like `<sdk>.git<git_hash>`. Set `NO_VERSION_LABEL=ON` to disable this behavior.
```
$ python -mbuild -w
...
Successfully built tilelang-0.1.6.post1+cu116.git0d4a74be-cp38-abi3-linux_x86_64.whl
```

where `<sdk>={cuda,rocm,metal}`. Specifically, when `<sdk>=cuda` and `CUDA_VERSION` is provided via env,
`<sdk>=cu<cuda_major><cuda_minor>`, similar with this part in pytorch.
Set `NO_TOOLCHAIN_VERSION=ON` to disable this.

### Run-time environment variables

<!-- TODO: tvm -->

## IDE Configs

Building tilelang locally will automatically `compile_commands.json` file in `build` dir.
VSCode with clangd and [clangd extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) should be able to index that without extra configuration.

## Compile cache

`ccache` will be automatically used if found.

## Repairing wheels

If you plan to use your wheel in other environment,
it's recommend to use auditwheel (on Linux) or delocate (on Darwin)
to repair them.

## Faster rebuild for developers

`pip install` introduces extra [un]packaging and takes ~30 sec to complete,
even if no source change.

Developers who needs to recompile frequently could use:

```bash
pip install -r requirements-dev.txt
pip install -e . -v --no-build-isolation

cd build; ninja
```

When running in editable/developer mode,
you'll see logs like below:

```console
$ python -c 'import tilelang'
2025-10-14 11:11:29  [TileLang:tilelang.env:WARNING]: Loading tilelang libs from dev root: /Users/yyc/repo/tilelang/build
```
