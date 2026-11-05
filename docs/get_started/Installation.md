# Installation Guide

TileLang uses a matched set of three distributions:

- `tilelang` owns the DSL, JIT adapters, device headers and Python extensions.
- `apache-tvm-tirx` owns `tvm_tirx`, the combined TileLang/TIRx compiler and its device runtimes.
- `apache-tvm-ffi` owns the shared object system, registry and FFI library.

The source manifest fixes both dependency versions. A normal upstream TIRx wheel
has no TileLang compiler registrations and cannot replace the matched provider.
Compiler extensions import `tvm_tirx` directly and use its native `Function`,
typed tensor and statement APIs.

## Installing with pip

Install from a channel or wheel directory containing the complete matched set:

```bash
python -m pip install --find-links /path/to/wheelhouse tilelang
```

A prebuilt TileLang wheel selects its provider and FFI versions automatically.
The FFI wheel must match the Python interpreter and platform; the provider is
independent of the Python ABI, and the frontend uses the stable ABI. Keep all
three distributions when copying a wheel bundle to another machine.

Source builds in this rebase produce source-specific dependency versions. They
do not publish those versions to PyPI. Installing `tilelang` from PyPI selects
whatever release is available there, not necessarily this source revision.
For this revision, build the bundle below or use its supplied wheel directory.

Verify package ownership and the actual loaded libraries outside the checkout:

```bash
python /path/to/tilelang/maint/scripts/verify_native_install.py \
  --import-order tilelang-first --output install-tilelang-first.json
python /path/to/tilelang/maint/scripts/verify_native_install.py \
  --import-order tirx-first --output install-tirx-first.json
```

The verifier also accepts `--expected-source /path/to/native-stage/manifest.json`.
With LLVM and Torch available, `--execute-cpu` checks a dynamic-shape TileLang
kernel numerically. Source generation alone does not validate GPU execution.

### Installing on AMD GPUs (ROCm)

Install a ROCm build of PyTorch from the channel matching the machine's runtime
before installing the matched bundle:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/rocm7.0
python -m pip install --find-links /path/to/wheelhouse tilelang
```

Linux distribution builds can contain CUDA, ROCm and Ascend emitters and runtime
stubs together. A kernel still needs its backend's real runtime, compiler and
device. ROCm execution requires `hipcc` and the matching ROCm installation.

## Building from Source

Use Python 3.10 or later, `uv`, CMake 3.26 or later, Ninja and a C++20 compiler.
LLVM development libraries and `llvm-config` are needed when enabling LLVM.
Initialize the submodules from the selected TileLang revision:

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
uv venv
source .venv/bin/activate
```

On Windows, activate `.venv\Scripts\activate` in a Visual Studio Developer
Command Prompt. Install Visual Studio Build Tools with the C++ toolchain. The
Windows ARM64 CUDA preview additionally requires its matching CUDA toolkit and
PyTorch preview; use the versions selected by the Windows ARM64 CI matrix.

Stage the exact compiler sources and bootstrap the fixed FFI before installing
development requirements. It may not be available on a public package index:

```bash
python maint/scripts/build_native_artifacts.py stage --destination .native-dev-stage
python maint/scripts/build_native_artifacts.py build \
  --stage .native-dev-stage --output native-dev-dist --frontend uv \
  --role ffi --wheel-only --install --jobs 4
uv pip install --find-links native-dev-dist -r requirements-dev.txt
```

Then build and install the provider and frontend in dependency order:

```bash
python maint/scripts/build_native_artifacts.py build \
  --stage .native-dev-stage --output native-dev-dist --frontend uv \
  --role provider --role frontend --wheel-only --install --jobs 4
```

`--python` selects a different target environment. `--install` modifies only that
environment. Omit `--wheel-only` to build each sdist and rebuild its wheel from
that archive. Commands, status and artifact hashes are saved with the output.
Staging directories are immutable inputs: choose a new stage after source edits,
and a new output directory when rebuilding a role or changing the source digest.

### With host CUDA toolchain

Put the desired toolkit on `PATH`, or set `CUDA_HOME`/`CUDAToolkit_ROOT` before
building the provider. For example:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export USE_CUDA=ON
```

### With pip-provided CUDA toolchain (no host CUDA required)

On supported platforms, install the toolkit packages in the selected environment
and point `WITH_PIP_CUDA_TOOLCHAIN` at their toolkit directory before building:

```bash
uv pip install "nvidia-cuda-nvcc>=13" "nvidia-cuda-cccl>=13" "nvidia-cuda-nvrtc>=13"
```

`cmake/FindPipCUDAToolkit.cmake` documents discovery and Windows architecture
handling. Runtime driver libraries must come from the machine's GPU driver.

### Working from Source via `PYTHONPATH` (Recommended for Developers)

Use an editable frontend over the matched installed provider:

```bash
uv pip install --no-deps -e .
```

This exposes Python source edits immediately while keeping one compiler, registry
and FFI library. An explicit `PYTHONPATH` pointing at the checkout has the same
compiler-identity requirement. Do not add an old TVM checkout, its Python package,
or an old `libtvm` directory to the environment.

A C++ or TIRx source edit changes the provider identity. Rebuild the matched
provider and frontend before importing again; see the developer rebuild section.
The development-only `TILELANG_NATIVE_SOURCE_OVERRIDE` creates partial compilers
for isolated tests. Such compilers deliberately cannot be installed or packaged.

### Building with Customized TVM Path

`TVM_ROOT` for the old compiler is obsolete. Changes to the new compiler belong
in the versioned `3rdparty/tvm-tirx` dependency and must be included in the source
manifest. Its FFI gitlink must match the pinned source in
`tilelang_packaging/compiler_manifest.py`. Build a new matched provider after
changing compiler sources; installing an unrelated compiler wheel is insufficient.

## Install Using Docker

Run the same matched-build commands inside a development container that supplies
the selected device SDK and a C++20 compiler. NVIDIA CUDA development images
provide `nvcc`; ROCm development images provide `hipcc`. The host supplies the
GPU driver and device access. A container build does not establish support for a
device absent from the machine.

### ROCm container build (gfx942/gfx950)

Keep the container's working ROCm Torch installation. Use a dedicated environment
(or a venv with system site packages when intentionally reusing Torch), initialize
the selected submodules, and set these flags before the matched provider build:

```bash
export USE_CUDA=OFF
export USE_ROCM=ON
export ROCM_PATH=/opt/rocm
```

Enable LLVM with the container's supported `llvm-config` when CPU compilation is
needed. The frontend and provider still use the fixed FFI built in the first
stage; do not install a separate old FFI version or disable dependency checks to
make an incompatible package load.

## Install with Nightly Version

A nightly channel must provide the complete matched wheel set:

```bash
python -m pip install tilelang --find-links https://tile-ai.github.io/whl/nightly
```

Availability of a source-specific version depends on publication to that channel.
For a local revision, use its locally built wheel directory instead.

## Install Configs

### Build-time environment variables

Set backend options before the provider build, or pass
`--cmake-define USE_LLVM=/path/to/llvm-config` to `build_native_artifacts.py build`.

| Option | Meaning |
| --- | --- |
| `USE_CUDA` | Enable CUDA; a toolkit is required at build time. Auto-detected on supported hosts. |
| `USE_ROCM` | Enable ROCm. Linux can use vendored HIP headers and runtime stubs for building without a ROCm SDK. |
| `USE_ASCEND` | Enable Ascend. Linux can build its emitters and runtime stubs without CANN. |
| `USE_METAL` | Enable Metal runtime support; defaults on for macOS. |
| `USE_LLVM` | Enable LLVM codegen; defaults off. Use `ON` or the path to `llvm-config`. |
| `WITH_PIP_CUDA_TOOLCHAIN` | Select a pip-installed CUDA toolkit directory. |
| `CMAKE_BUILD_PARALLEL_LEVEL` | Bound native build concurrency; the bundle scripts also accept `--jobs`. |
| `NO_VERSION_LABEL` | Control the frontend version label; it does not remove the exact provider/FFI pins. |
| `NO_TOOLCHAIN_VERSION` | Omit the frontend's toolkit version label. |

Linux defaults include ROCm and Ascend emitters. Set
`USE_CUDA=OFF USE_ROCM=OFF USE_ASCEND=OFF` for a build without those backends.
Hardware support and execution must be tested independently for each backend.

### Run-time environment variables

See `tilelang/env.py` for JIT, cache, target and toolchain options. Library loading
checks the installed provider identity. `TVM_TIRX_LIBRARY_PATH` is a development
library-search setting, not a way to substitute an incompatible compiler.

## Other Tips

### IDE Configs

Provider CMake builds emit `compile_commands.json` in their build directory.
Point clangd at that file to use the actual TIRx headers and build definitions.

### Compile Cache

Kernel caches include the native provider identity. Caches from the old compiler
or another provider revision are invalidated. `ccache` can reduce repeated native
compilation; it does not replace the source manifest or validation.

### Repairing Wheels

`maint/scripts/build_native_wheels.py` builds a complete portable bundle using
cibuildwheel on Linux/macOS and matching FFI wheels plus native builds on Windows:

```bash
python maint/scripts/build_native_wheels.py \
  --stage .native-release-stage --output wheelhouse --jobs 4
```

It builds the required CPython FFI variants before testing the provider/frontend.
Wheel repair keeps FFI owned by its own distribution and Z3 by its package. The
script does not publish artifacts. Public release versioning, access to the
provider/FFI publication channels and availability of the exact dependencies
must be established before publishing a frontend release.

(faster-rebuild-for-developers)=

### Faster Rebuild for Developers

Pure Python changes only need the editable frontend. After native source edits,
create a new stage and rebuild its provider and frontend; the previously installed
fixed FFI can be reused:

```bash
python maint/scripts/build_native_artifacts.py stage --destination .native-rebuild-stage
python maint/scripts/build_native_artifacts.py build \
  --stage .native-rebuild-stage --output native-rebuild-dist --frontend uv \
  --role provider --role frontend --wheel-only --install --jobs 4
uv pip install --no-deps -e .
```

Keep the prior bundle to roll back both compiler and frontend together. Run the
installation verifier from outside the checkout after a native rebuild, and run
the tests relevant to the changed passes, adapters and backends.
