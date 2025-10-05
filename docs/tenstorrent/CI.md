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
3. Install system dependencies: cmake, ninja, llvm-dev, libpolly-dev
4. Install Python dependencies from requirements-test.txt
5. **TVM Build Caching:**
   - Generate cache key based on TVM submodule commit hash
   - Restore cached TVM build artifacts if available (uses `actions/cache/restore@v4`)
   - Caches: `build/libtvm*.so` and `build/3rdparty/`
   - Save TVM artifacts after build completes (uses `actions/cache/save@v4` with `if: always()`)
   - Cache is saved even if job fails, preventing redundant TVM rebuilds
   - Only rebuilds TVM when the submodule is updated
6. Build TileLang with LLVM backend
   - Uses Ninja build system
   - Limited to 2 parallel jobs to avoid OOM on GitHub runners
   - LLVM backend is sufficient for CPU-only testing
7. Install TileLang in development mode
   - Sets `USE_LLVM=true` to enable LLVM backend
   - setup.py checks for nvcc availability before trying to use it
   - Gracefully skips CUDA version detection if nvcc is not found
8. Run Tenstorrent target registration tests
9. Run all Tenstorrent Python tests (CPU-only)

**Caching Strategy:**
- **TVM build artifacts:** Keyed by TVM submodule commit + OS
  - Dramatically reduces build time (TVM build is expensive)
  - Only invalidates when TVM submodule is updated
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

- **First run:** ~30-40 minutes (builds TVM from scratch)
- **Subsequent runs (no TVM changes):** ~5-10 minutes (TVM cache hit)
- **Cache storage:** GitHub Actions provides up to 10GB cache per repository
- **Cache eviction:** GitHub evicts caches not accessed in 7 days

## Future Improvements

Potential optimizations:
- Add CUDA build and GPU testing when needed (will require NVIDIA container or GPU runners)
- Custom Docker image with pre-built TVM (eliminates TVM build entirely)
- Parallel test execution with pytest-xdist
- Separate workflow for expensive builds (only on main/release branches)
