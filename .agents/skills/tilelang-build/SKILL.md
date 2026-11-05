---
name: tilelang-build
description: Repository-specific build, rebuild, install, and test instructions for tilelang. Use when working in the tilelang repository and the correct commands are needed for building from source, reinstalling after changes, or running project tests.
---

# Build & Install

TileLang uses three matched distributions: the `tilelang` frontend, an
`apache-tvm-tirx` provider containing the combined compiler and device runtimes,
and the fixed `apache-tvm-ffi`. Follow `docs/get_started/Installation.md` for
toolchain and platform details. An ordinary upstream TIRx wheel does not include
TileLang registrations, and installing the frontend alone does not build them.

## Initial Source Build

Use Python 3.10+, `uv`, CMake 3.26+, Ninja and a C++20 compiler. Start in the
selected checkout and an activated dedicated environment:

```bash
git submodule update --init --recursive
uv venv
source .venv/bin/activate
python maint/scripts/build_native_artifacts.py stage --destination .native-dev-stage
python maint/scripts/build_native_artifacts.py build \
  --stage .native-dev-stage --output native-dev-dist --frontend uv \
  --role ffi --wheel-only --install --jobs 4
uv pip install --find-links native-dev-dist -r requirements-dev.txt
python maint/scripts/build_native_artifacts.py build \
  --stage .native-dev-stage --output native-dev-dist --frontend uv \
  --role provider --role frontend --wheel-only --install --jobs 4
```

The FFI bootstrap precedes requirements installation because its source-specific
version may not be available on a package index. `--python` selects the target
environment. A build without `--wheel-only` creates sdists and rebuilds wheels
from them. Build logs and artifact hashes stay in the output directory.

Backend switches apply to the provider build:

| Setting | Purpose |
| --- | --- |
| `USE_CUDA=ON/OFF` | Enable/disable CUDA; ON needs a supported toolkit. |
| `USE_ROCM=ON/OFF` | Select ROCm/HIP emitters and runtime. |
| `USE_ASCEND=ON/OFF` | Select Ascend emitters and runtime. |
| `USE_METAL=ON/OFF` | Select Metal; normally enabled on macOS. |
| `--cmake-define USE_LLVM=/path/to/llvm-config` | Enable the selected LLVM toolchain. |
| `SKBUILD_CMAKE_BUILD_TYPE=Debug` | Select a Debug PEP 517 native build. |

Linux defaults include ROCm and Ascend stubs. Their presence permits imports
without those devices; executing kernels still needs the corresponding SDK,
runtime and hardware. Debug logging additionally needs the `TVM_LOG_DEBUG=1`
C++ preprocessor definition; see `docs/tutorials/logging.md`.

## Editable Frontend and Native Rebuilds

After installing the matched bundle, enable editable Python imports:

```bash
uv pip install --no-deps -e .
```

Python-only edits are reflected immediately. Importing the local checkout via
`PYTHONPATH` still requires the matching installed provider. The loader no
longer searches old `build/tvm/` or `libtilelang` compiler artifacts.

A native source edit changes the provider identity. Use a fresh immutable stage
and output directory, reuse the already installed fixed FFI, and rebuild both
provider and frontend:

```bash
python maint/scripts/build_native_artifacts.py stage --destination .native-rebuild-stage
python maint/scripts/build_native_artifacts.py build \
  --stage .native-rebuild-stage --output native-rebuild-dist --frontend uv \
  --role provider --role frontend --wheel-only --install --jobs 4
uv pip install --no-deps -e .
```

Use a new stage/output name for each later native revision. `ccache` can reuse
unchanged compilation inputs. Direct CMake builds and partial-source overrides
are useful for isolated compiler work, but do not install or validate a matched
bundle automatically. `TILELANG_NATIVE_SOURCE_OVERRIDE` builds cannot be installed.

From outside the checkout, verify installed ownership and library identity:

```bash
python /path/to/tilelang/maint/scripts/verify_native_install.py \
  --import-order tilelang-first --output install-tilelang-first.json
python /path/to/tilelang/maint/scripts/verify_native_install.py \
  --import-order tirx-first --output install-tirx-first.json
```

## Running Tests

Choose tests for the changed subsystem and available backend. Many device tests
need a GPU; construction and source-generation results have a narrower scope.

```bash
python -m pytest testing/python/language/test_tilelang_language_copy.py -x
python -m pytest testing/python/metal/ -x
```

Some native regression tests need test-only C++ bridges. Build their matching
provider with `--cmake-define TILELANG_BUILD_NATIVE_TESTS=ON` and run ordinary
pytest. `testing/conftest.py` makes the shared native Python fixtures available;
tests using a C++ bridge require its registrations in that provider. Production
providers leave this option OFF and do not ship test registry entries.
