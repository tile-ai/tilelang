# Tenstorrent Backend CI

This document describes the CI setup for the Tenstorrent backend in TileLang.

## Overview

The Tenstorrent backend CI is defined in `.github/workflows/tenstorrent-ci.yml` and runs on:
- Pull requests that modify Tenstorrent-related files
- Pushes to `main` and `ws1-**` branches

## Jobs

### 1. Lint and Format (`lint-and-format`)

**Environment:** Ubuntu runner with Python 3.10

**Purpose:** Ensure code formatting and style consistency

**Steps:**
1. Checkout repository with submodules
2. Set up Python with pip caching (caches `requirements-lint.txt` dependencies)
3. Install lint dependencies: yapf, ruff, codespell, clang-format
4. Run `format.sh` to check formatting compliance
   - If formatting issues are found, the job fails and shows the diff

**Caching:**
- Pip packages are cached based on `requirements-lint.txt` hash
- Subsequent runs with unchanged dependencies skip pip installation

### 2. Build and Test (`build-and-test`)

**Environment:** Ubuntu runner with Python 3.10

**Purpose:** Build TileLang with LLVM backend and run Tenstorrent tests

**Note:** Currently builds with LLVM backend (not CUDA) since we only run CPU tests at this stage. This keeps the CI lightweight and fast. GPU/CUDA testing will be added in future when needed.

**Steps:**
1. Checkout repository with submodules
2. Set up Python with pip caching (caches `requirements-test.txt` dependencies)
3. Install system dependencies: cmake, ninja, llvm, build-essential, libedit-dev, libxml2-dev, zlib1g-dev
4. Install Python dependencies from requirements-test.txt
5. **Enable ccache:**
   - Uses `hendrikmuhs/ccache-action` for compiler caching
   - Cache key based on CMakeLists.txt hash + OS + version
   - Max size: 2G
   - Creates symlinks for automatic use by CMake
6. **TVM Build Caching:**
   - Generate cache key based on TVM submodule commit hash
   - Restore cached TVM build artifacts if available (uses `actions/cache/restore@v4`)
   - Caches: `build/tvm/` (contains libtvm*.so), `build/libtilelang*.so`, and `build/3rdparty/`
   - Save TVM artifacts after build completes (uses `actions/cache/save@v4` with `if: always()`)
   - Cache is saved even if job fails, preventing redundant TVM rebuilds
   - Only rebuilds TVM when the submodule is updated
7. Build TileLang with LLVM backend (ccache-enabled)
   - Uses Ninja build system with ccache as compiler launcher
   - Limited to 2 parallel jobs to avoid OOM on GitHub runners
   - LLVM backend is sufficient for CPU-only testing
   - Uses system LLVM packages instead of downloading LLVM 10.0.1
8. Install TileLang and TVM Python packages
   - Install TVM Python package from `3rdparty/tvm/python` with `TVM_LIBRARY_PATH` set
   - Install TileLang with `USE_LLVM=true` to enable LLVM backend
   - setup.py checks for nvcc availability before trying to use it
   - Gracefully skips CUDA version detection if nvcc is not found
9. Print ccache statistics (with availability check)
10. Run Tenstorrent target registration tests
    - Sets `LD_LIBRARY_PATH` to include `build/tvm` for TVM library discovery
    - Continue-on-error enabled for graceful handling
11. Run all Tenstorrent Python tests (CPU-only)
    - Sets `LD_LIBRARY_PATH` to include `build/tvm` for TVM library discovery
    - Continue-on-error enabled for graceful handling

**Caching Strategy:**
- **ccache (compiler cache):** Keyed by CMakeLists.txt hash + OS + version
  - Caches compiled object files for fast recompilation
  - 2G maximum size
- **TVM build artifacts:** Keyed by TVM submodule commit + OS
  - Dramatically reduces build time (TVM build is expensive)
  - Only invalidates when TVM submodule is updated
  - Saved even on job failure to prevent rebuilding on retry
- **Pip packages:** Keyed by requirements-test.txt hash
  - Reuses cached pytest and other test dependencies

### 3. Static Analysis (`static-analysis`)

**Environment:** Ubuntu runner with Python 3.10

**Purpose:** Type checking with mypy

**Steps:**
1. Checkout repository
2. Set up Python with pip caching (caches `requirements-mypy.txt` dependencies)
3. Install mypy from requirements-mypy.txt
4. Run mypy on `tilelang/engine/tt/` (currently set to continue-on-error)

**Caching:**
- Pip packages are cached based on `requirements-mypy.txt` hash
- Ensures consistent caching behavior across CI runs

## Caching Summary

The CI uses multiple layers of caching for efficiency:

| Job | What's Cached | Cache Key | Benefit |
|-----|---------------|-----------|---------|
| lint-and-format | Pip packages | requirements-lint.txt hash | Fast linter installation |
| build-and-test | TVM build artifacts | TVM submodule commit + OS | Avoid rebuilding TVM (~30+ min), saved even on failure |
| build-and-test | ccache compiler cache | CMakeLists.txt hash + OS + version | Fast recompilation of unchanged files |
| build-and-test | Pip packages | requirements-test.txt hash | Fast pytest install |
| static-analysis | Pip packages | requirements-mypy.txt hash | Fast mypy installation |

## Running Locally

To ensure your changes will pass CI:

```bash
# Run formatting checks
bash format.sh

# If format.sh makes changes, review and commit them
git diff
git add .
git commit -m "Apply formatting"

# Run tests (requires TileLang built with TVM)
cd testing/python/tt
pytest test_target_registration.py -v
```

## Triggering CI

CI runs automatically on:
- Pull requests modifying:
  - `tilelang/engine/tt/**`
  - `testing/python/tt/**`
  - `tilelang/utils/target.py`
  - `.github/workflows/tenstorrent-ci.yml`
- Pushes to `main` or `ws1-**` branches

## Performance Notes

- **First run:** ~6-7 minutes (builds TVM from scratch with ccache)
- **Subsequent runs (TVM cache hit):** ~30-60 seconds (skips TVM build, uses ccache for incremental builds)
- **Cache storage:** GitHub Actions provides up to 10GB cache per repository
- **Cache eviction:** GitHub evicts caches not accessed in 7 days
- **ccache effectiveness:** Dramatically reduces compilation time for unchanged files
- **TVM cache effectiveness:** Eliminates ~5-6 minutes of TVM rebuild when submodule unchanged

## Key Design Decisions

1. **System LLVM vs Downloaded LLVM:** Uses system LLVM packages (installed via apt) instead of downloading LLVM 10.0.1. This avoids compatibility issues with newer Ubuntu versions, which do not include `libtinfo.so.5` by default—causing runtime linking errors when using the downloaded LLVM 10.0.1 binaries.

2. **Separate TVM Python Installation:** TVM Python package is installed separately before TileLang to ensure proper library path configuration.

3. **LD_LIBRARY_PATH for Tests:** Tests require `LD_LIBRARY_PATH` to be set to `build/tvm` so Python can find the TVM shared libraries at runtime.

4. **Cache Split (Restore/Save):** Using separate `actions/cache/restore` and `actions/cache/save` with `if: always()` ensures TVM cache is saved even when the job fails, preventing redundant rebuilds on retry.

5. **Continue-on-error for Tests:** Tests are marked with `continue-on-error: true` because the Tenstorrent target registration in TVM is incomplete. As a result, tests are expected to fail until the backend implementation and target registration are finished.

## Future Improvements

Potential optimizations:
- Add CUDA build and GPU testing when needed (will require NVIDIA container or GPU runners)
- Custom Docker image with pre-built TVM (eliminates TVM build entirely)
- Parallel test execution with pytest-xdist
- Separate workflow for expensive builds (only on main/release branches)
- Remove continue-on-error once Tenstorrent backend is fully implemented
