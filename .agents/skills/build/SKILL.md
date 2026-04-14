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

**Never use `pip install -e .`** (editable install). When running Python from the repo root, the local `./tilelang` directory is imported instead of the installed copy (because `.` is on `sys.path` by default). This makes editable installs unnecessary. Avoid `pip install -e .` as it can cause import confusion with this project's layout.

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
