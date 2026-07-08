---
name: tilelang-build
description: Repository-specific build, rebuild, install, and test instructions for tilelang. Use when working in the tilelang repository and the correct commands are needed for building from source, reinstalling after changes, or running project tests.
---

# Build & Install

## Installing / Rebuilding tilelang

The standard way to build and install:

```bash
pip install .
```

Or with verbose output for debugging build issues:

```bash
pip install . -v
```

`uv pip install .` also works if `uv` is available but is not required.

Build dependencies are declared in `pyproject.toml` and resolved automatically during `pip install .`.

If `ccache` is available, repeated builds only recompile changed C++ files.

## Alternative: Development Build with `--no-build-isolation`

If you need faster iteration (e.g. calling `cmake` directly to recompile C++ without re-running the full pip install), install build dependencies first:

```bash
pip install -r requirements-dev.txt
pip install --no-build-isolation .
```

After this, you can invoke `cmake --build build` directly to recompile only changed C++ files. This is useful when iterating on C++ code.

## Alternative: cmake + PYTHONPATH (recommended for C++ development)

For the fastest C++ iteration, bypass pip entirely and drive cmake directly:

```bash
# Configure (auto-detects CUDA; git submodules are initialised automatically)
cmake -S . -B build

# Build
cmake --build build -j$(nproc)

# Make the local tilelang package importable
export PYTHONPATH=$(pwd):$PYTHONPATH
```

After the initial configure, recompiling is just `cmake --build build -j$(nproc)`. The runtime automatically discovers native libraries from `build/lib/` when it detects a dev checkout (see `tilelang/env.py`).

Useful cmake options:

| Flag | Purpose |
|------|---------|
| `-DUSE_CUDA=ON/OFF` | Enable/disable CUDA backend (ON by default) |
| `-DUSE_ROCM=ON` | Enable ROCm/HIP backend |
| `-DUSE_METAL=ON` | Enable Metal backend (default on macOS) |
| `-DCMAKE_BUILD_TYPE=Debug` | Debug build with `TVM_LOG_DEBUG` enabled |

## Editable Installs

`pip install -e .` is a supported development install, and is what `README.md`, `CONTRIBUTING.md`, and `docs/get_started/Installation.md` recommend for development:

```bash
pip install -r requirements-dev.txt
pip install -e . -v --no-build-isolation
```

Notes specific to this repo's layout:

- When Python is run from the repo root, the local `./tilelang` directory is imported instead of any copy installed into `site-packages` (because the repo root is on `sys.path` ahead of `site-packages`). This is by design: `tilelang/env.py` detects a dev checkout (when `3rdparty/` is not inside the package dir) and loads native libraries from `build/lib/` and `build/tvm/`, logging `Loading tilelang libs from dev root: <repo>/build`.
- Because of the above, both approaches resolve imports to the local `./tilelang` when run from the repo root. Use `pip install -e .` when you want pip to manage package metadata/dependencies; use `PYTHONPATH=$(pwd)` only for the lighter-weight import-only workflow.
- For pure C++ iteration (no Python metadata needed), the `cmake + PYTHONPATH` flow above is faster and avoids re-running pip entirely.

## Running Tests

Most tests require a GPU.

```bash
python -m pytest testing/python/ -x
```

Run a specific test file or test case:

```bash
python -m pytest testing/python/language/test_tilelang_language_copy.py -x
python -m pytest testing/python/language/test_tilelang_language_copy.py -x -k "test_name"
```

For Metal-specific tests (requires macOS with Apple Silicon):

```bash
python -m pytest testing/python/metal/ -x
```
