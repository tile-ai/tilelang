# Installation Guide

## Installing with pip

**Prerequisites for installation via wheel or PyPI:**

- **glibc**: 2.28 (Ubuntu 20.04 or later)
- **Python Version**: >= 3.8
- **CUDA Version**: 12.0 <= CUDA < 13

The easiest way to install tilelang is directly from PyPI using pip. To install the latest version, run the following command in your terminal:

```bash
pip install tilelang
```

Alternatively, you may choose to install tilelang using prebuilt packages available on the Release Page:

```bash
pip install tilelang-0.0.0.dev0+ubuntu.20.4.cu120-py3-none-any.whl
```

To install the latest version of tilelang from the GitHub repository, you can run the following command:

```bash
pip install git+https://github.com/tile-ai/tilelang.git
```

After installing tilelang, you can verify the installation by running:

```bash
python -c "import tilelang; print(tilelang.__version__)"
```

## Building from Source

**Prerequisites for building from source:**

- **Operating System**: Linux
- **Python Version**: >= 3.8
- **CUDA Version**: >= 10.0

If you prefer Docker, please skip to the [Install Using Docker](#install-using-docker) section. This section focuses on building from source on a native Linux environment.

First, install the OS-level prerequisites on Ubuntu/Debian-based systems using the following commands:

```bash
apt-get update
apt-get install -y python3 python3-dev python3-setuptools gcc zlib1g-dev build-essential cmake libedit-dev
```

Then, clone the tilelang repository and install it using pip. The `-v` flag enables verbose output during the build process.

> **Note**: Use the `--recursive` flag to include necessary submodules. Tilelang currently depends on a customized version of TVM, which is included as a submodule. If you prefer [Building with Existing TVM Installation](#using-existing-tvm), you can skip cloning the TVM submodule (but still need other dependencies).

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
pip install . -v
```

If you want to install tilelang in development mode, you can use the `-e` flag so that any changes to the Python files will be reflected immediately without reinstallation.

```bash
pip install -e . -v
```

> **Note**: changes to C++ files require rebuilding the tilelang C++ library. See [Faster Rebuild for Developers](#faster-rebuild-for-developers) below. A default `build` directory will be created if you use `pip install`, so you can also directly run `make` in the `build` directory to rebuild it as [Working from Source via PYTHONPATH](#working-from-source-via-pythonpath) suggested below.

(working-from-source-via-pythonpath)=

### Working from Source via `PYTHONPATH` (Recommended for Developers)

If you prefer to work directly from the source tree via `PYTHONPATH` instead of using pip, make sure the native extension (`libtilelang.so`) is built first:

```bash
mkdir -p build
cd build
cmake .. -DUSE_CUDA=ON
make -j
```

We also recommend using `ninja` to speed up compilation:

```bash
cmake .. -DUSE_CUDA=ON -G Ninja
ninja
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

(using-existing-tvm)=

### Building with Customized TVM Path

If you already have a TVM codebase, use the `TVM_ROOT` environment variable to specify the location of your existing TVM repository when building tilelang:

```bash
TVM_ROOT=<your-tvm-repo> pip install . -v
```

> **Note**: This will still rebuild the TVM-related libraries (stored in `TL_LIBS`). And this method often leads to some path issues. Check `env.py` to see some environment variables which are not set properly.

(install-using-docker)=

## Install Using Docker

For users who prefer a containerized environment with all dependencies pre-configured, tilelang provides Docker images for different CUDA versions. This method is particularly useful for ensuring consistent environments across different systems.

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

4. **Access the Container and Verify Installation**:

```bash
docker exec -it tilelang_b200 /bin/zsh
# Inside the container:
python -c "import tilelang; print(tilelang.__version__)"
```

### ROCm container build (gfx942/gfx950)

If you want a ready-to-use ROCm image that builds TileLang from source, use
`docker/Dockerfile.rocm`. This is the recommended path for a clean, reproducible
environment.

If you are already inside another ROCm container (for example, the `sglang`
image) and just need to rebuild TileLang in-place, follow the steps below.

If you are using the `sglang` ROCm container and need to build TileLang in it (for example on MI300 `gfx942` or MI355 `gfx950`), the build requires extra system libraries, Cython, and a valid `llvm-config`. The following steps match the build flow used in `sglang/docker/rocm.Dockerfile`:

```bash
# Inside the container (as root)
apt-get update && apt-get install -y --no-install-recommends \
  build-essential git wget curl ca-certificates gnupg \
  libgtest-dev libgmock-dev \
  libprotobuf-dev protobuf-compiler libgflags-dev libsqlite3-dev \
  python3 python3-dev python3-setuptools python3-pip \
  gcc libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
  cmake ninja-build pkg-config libstdc++6 \
  && rm -rf /var/lib/apt/lists/*

# Prefer the container venv (avoid system pip)
export PATH="/opt/venv/bin:${PATH}"

# Build GoogleTest static libs (Ubuntu package ships sources only)
cmake -S /usr/src/googletest -B /tmp/build-gtest -DBUILD_GTEST=ON -DBUILD_GMOCK=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/build-gtest -j"$(nproc)"
cp -v /tmp/build-gtest/lib/*.a /usr/lib/x86_64-linux-gnu/
rm -rf /tmp/build-gtest

# Keep setuptools < 80 (compat with some base images)
pip install --upgrade "setuptools>=77.0.3,<80" wheel cmake ninja scikit-build-core

# Locate ROCm llvm-config (install LLVM 18 if missing)
LLVM_CONFIG_PATH=""
for p in /opt/rocm/llvm/bin/llvm-config /opt/rocm/llvm-*/bin/llvm-config /opt/rocm-*/llvm*/bin/llvm-config; do
  if [ -x "$p" ]; then LLVM_CONFIG_PATH="$p"; break; fi
done
if [ -z "$LLVM_CONFIG_PATH" ]; then
  echo "ROCm llvm-config not found; installing LLVM 18..."
  curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh
  chmod +x /tmp/llvm.sh
  /tmp/llvm.sh 18
  LLVM_CONFIG_PATH="$(command -v llvm-config-18)"
  if [ -z "$LLVM_CONFIG_PATH" ]; then
    echo "ERROR: llvm-config-18 not found after install"
    exit 1
  fi
fi
export LLVM_CONFIG="$LLVM_CONFIG_PATH"
export PATH="$(dirname "$LLVM_CONFIG"):/usr/local/bin:${PATH}"

# Optional shim for tools that expect llvm-config-16
mkdir -p /usr/local/bin
printf "#!/usr/bin/env bash\nexec \"%s\" \"\$@\"\n" "$LLVM_CONFIG_PATH" > /usr/local/bin/llvm-config-16
chmod +x /usr/local/bin/llvm-config-16

# TVM Python bits need Cython (for system Python used by the build)
pip install --no-cache-dir "cython>=0.29.36,<3.0"

# Clone + build TileLang (ROCm)
# Default location: /opt/tilelang (adjust if you prefer a different path).
git clone --recursive https://github.com/tile-ai/tilelang.git /opt/tilelang
cd /opt/tilelang
git submodule update --init --recursive
export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=ON -DROCM_PATH=/opt/rocm -DLLVM_CONFIG=${LLVM_CONFIG}"

# Avoid pulling CUDA wheels / reinstalling torch by skipping dependency resolution.
# Assume torch is already installed in the container.
pip install -e . -v --no-build-isolation --no-deps

# Manually install required runtime deps when using --no-deps.
# Note: skip torch-c-dlpack-ext on ROCm (its wheel expects CUDA libs).
pip install "apache-tvm-ffi>=0.1.6" "z3-solver>=4.13.0"
# If you already installed torch-c-dlpack-ext and hit `libtorch_cuda.so` errors:
# pip uninstall -y torch-c-dlpack-ext

# If you hit Cython compile errors like `PyLong_SHIFT`/`digit` not declared,
# disable the stable ABI (abi3) for editable builds:
# export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=ON -DROCM_PATH=/opt/rocm -DLLVM_CONFIG=${LLVM_CONFIG} -DSKBUILD_SABI_VERSION="
# pip install -e . -v --no-build-isolation --no-deps

# Verify
python -c "import tilelang; print(tilelang.__version__)"
```

If you still want to use `pip install -e . -v --no-build-isolation` without `--no-deps`, pip will try to resolve TileLang dependencies and may download CUDA wheels (e.g., `nvidia_cudnn`, `nvidia_nvshmem`) and reinstall `torch`. To avoid that in ROCm containers, keep `--no-deps` and ensure required packages are already installed.

## Install with Nightly Version

For users who want access to the latest features and improvements before official releases, we provide nightly builds of tilelang.

```bash
pip install tilelang -f https://tile-ai.github.io/whl/nightly
# or pip install tilelang --find-links https://tile-ai.github.io/whl/nightly
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

Please refer to the `env.py` file for a full list of supported run-time environment variables.

## Other Tips

### IDE Configs

Building tilelang locally will automatically generate a `compile_commands.json` file in `build` dir.
VSCode with clangd and [clangd extension](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) should be able to index that without extra configuration.

### Compile Cache

The default path of the compile cache is `~/.tilelang/cache`. `ccache` will be automatically used if found.

### Repairing Wheels

If you plan to use your wheel in other environment,
it's recommended to use auditwheel (on Linux) or delocate (on Darwin)
to repair them.

(faster-rebuild-for-developers)=

### Faster Rebuild for Developers

`pip install` introduces extra [un]packaging and takes ~30 sec to complete,
even if no source change.

Developers who needs to recompile frequently could use:

```bash
pip install -r requirements-dev.txt

# For first time compilation
pip install -e . -v --no-build-isolation

# Or manually compile with cmake/ninja. Remember to set PYTHONPATH properly.
mkdir build
cd build
cmake .. -G Ninja
ninja

# Rebuild when you change the cpp code
cd build; ninja
```

When running in editable/developer mode,
you'll see logs like below:

```console
$ python -c 'import tilelang'
2025-10-14 11:11:29  [TileLang:tilelang.env:WARNING]: Loading tilelang libs from dev root: /Users/yyc/repo/tilelang/build
```
