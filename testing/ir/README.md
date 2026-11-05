# Native compiler regression tests

These tests use TileLang's native TIRx objects and the fixed companion compiler.
Several cases inspect private C++ analyses through the small bridges in
`testing/cpp/`. Build those bridges into the **same compiler** as TileLang. In a
dedicated environment, stage and install the matched FFI, test provider and
frontend:

```bash
python maint/scripts/build_native_artifacts.py stage --destination .native-test-stage
python maint/scripts/build_native_artifacts.py build \
  --stage .native-test-stage --output .native-test-artifacts \
  --frontend uv --wheel-only --install --jobs 4 \
  --cmake-define TILELANG_BUILD_NATIVE_TESTS=ON
uv run --no-sync python -m pytest \
  testing/ir/test_native_constraint_instances.py \
  testing/ir/test_native_tileop_access_regions.py \
  testing/python/cpu/test_async_width_contract.py
```

Keep the other CMake options appropriate for the selected backend. In particular,
Ascend analysis bridges require `USE_ASCEND=ON`; LLVM execution cases require a
configured LLVM backend. The shared fixtures are found by `testing/conftest.py`.
The commands use ordinary pytest, with no external Python runner or probe DSO.

For a direct CMake development build, first install the fixed FFI version used
by this checkout and configure the required backend options. Select the local
Python sources and libraries explicitly when running the tests; rebuilding with
CMake does not replace an already installed provider:

```bash
cmake -S . -B build -DTILELANG_BUILD_NATIVE_TESTS=ON
cmake --build build --target tvm_tirx_compiler --parallel 4
PYTHONPATH="$PWD:$PWD/3rdparty/tvm-tirx/python:$PWD/build/lib" \
TVM_TIRX_LIBRARY_PATH="$PWD/build/lib" \
  uv run --no-sync python -m pytest testing/ir/test_native_constraint_instances.py
```

The provider sdist includes the exact bridge sources in its source identity.
The option defaults to `OFF` for production providers; disable it and rebuild
the compiler before producing release artifacts. Optional independently built
probe libraries used during debugging must not be loaded into a compiler that
already contains the same test registrations.

Select hardware-specific cases explicitly. This directory also contains CUDA
tests for SM100/SM120, architecture-specific compilation checks, and platform
runtime cases. Constructing IR or emitting source does not validate device
execution, and the presence of a CUDA device does not satisfy every architecture
requirement. Existing numerical assertions and platform requirements apply to
these cases unchanged.
