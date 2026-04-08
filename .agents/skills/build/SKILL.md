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

## Editable Installs

**Never use `pip install -e .`** (editable install). When you install from source with `pip install .`, the local `./tilelang` directory is imported (not the installed copy). This is intentional and makes development convenient without editable mode. Editable installs break this assumption and cause import confusion.

## Running Tests

```bash
python -m pytest testing/python/ -x
```

For Metal-specific tests (requires macOS with Apple Silicon):

```bash
python -m pytest testing/python/metal/ -x
```
