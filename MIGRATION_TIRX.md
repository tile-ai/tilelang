# Standalone TIRx migration

## Status and dependency contract

TileLang now targets **unmodified upstream TIRx source** at the fixed revision
`9af5544df8553c35b6a85927cd2589f0a596f2fc` (`apache-tvm-tirx==0.1.0a2`).
`TVM_TIRX_COMMIT` and `TVM_TIRX_VERSION` pin that source and package version.
The former `0.1.0a2+tilelang.1` integration patches are no longer required.

- There is no TVM/TIRx checkout or submodule under TileLang's `3rdparty/`.
- Python imports `tvm_tirx` and `tvm_ffi`. `tilelang.tvm` is a local adapter;
  it does not create a global `tvm` module alias.
- The TileLang wheel does not bundle TIRx compiler, runtime, or FFI libraries.
- Native IR objects and their Python classes remain owned by TIRx/FFI.
- TileLang does not replace native FFI registrations, type hooks, transform
  factories, or implementation symbols, and does not include dependency `.cc`
  files in its build.

The clean-source build, migration regression suites, and refreshed independent
wheel checks have passed. The results below are from this zero-patch version,
not the earlier patched integration. This is not a public-release or
all-hardware qualification claim.

### Binary compatibility is still a build requirement

Zero source patches does **not** mean any prebuilt upstream wheel exports the
private C++ ABI used by TileLang. Build the pinned dependency with
`testing/tools/prepare_tirx.py`, or provide a wheel built with the equivalent
exports and backend configuration.

`cmake/tirx_extension_exports.cmake` owns symbol visibility. Source integration
configures existing dependency targets; independent dependency builds use the
standard `CMAKE_PROJECT_tvm_tirx_INCLUDE` hook. This changes build settings, not
TIRx source or symbol implementations. The native default visibility policy
remains unchanged in the upstream checkout.

`src/tirx_compat/` contains declarations for the pinned internal helper and
runtime factory ABI. An installed dependency need not ship its private headers,
and TileLang does not add the dependency's `src/` directory to its include path.
These declarations must be reviewed when changing the source pin. This is a
pinned compatibility surface, not a stable upstream C++ SDK guarantee.

## Extension ownership

| Concern | TileLang-owned implementation | Native functionality retained |
| --- | --- | --- |
| Expression-bearing attributes, lexical metadata, definition renewal | `src/compat/metadata.h`, `src/op/metadata.cc`, `src/op/check_well_formed.cc` | Native `Var`, `Call`, `For`, `RegionStmt`, and `Function` |
| Exact cast/custom dtype/storage/serialization services | `src/op/dtype.cc`, `src/op/prim.cc`, `src/op/serialization.cc`, `tilelang._compat.ir` | Native `PrimType`, `Cast`, reflection, and tensor/runtime types |
| Symbolic query normalization and context cleanup | `src/transform/common/analyzer.cc`, `src/compat/context.h`, `tilelang._compat.sym` | Native `sym::Analyzer` state and solvers |
| Extended transforms | `src/transform/tirx/`, `tilelang.transform._tirx_passes` | Existing native passes where their semantics suffice |
| Packed parameters/results and output storage | `src/transform/make_packed_api.cc`, downstream packed binder, `src/runtime/packed_result.cc` | Native runtime and TVM FFI |
| C-family code generation | `src/backend/common/codegen/tirx/` and backend-owned codegen directories | Native module factories and runtime |
| LLVM lowering | TileLang lowers its constructs before invoking native LLVM | Native LLVM implementation |
| CUDA Python codegen | `tilelang._compat.cuda_registry` and related adapters | Read-only reuse of the native codegen registry |

TileLang-owned transforms register under `tl._tirx_passes.*`; CUDA extensions
register under `tl.codegen.cuda.*`. Reuse of a native pass is explicit rather
than a monkey patch of `tvm_tirx.tirx.transform`.

The native pointer-value-type rewriter is also reused. The TileLang adapter in
`src/transform/storage_rewrite.cc` conservatively preserves storage, parameter,
and let element types when extension metadata refers to tensors or pointers.
Native vector-storage rewriting would otherwise leave those references and
their element units stale. Per-access vectorization remains enabled; this is
an explicit optimization boundary, not a replacement of native type hooks.

An access-region annotation is not a pointer's actual base address. Safe-memory
legalization preserves the operand's offset and guards its transfer extent,
including pointers that have already been lowered before entering the pipeline.

### Explicit adapters, not changed native semantics

Use `tilelang._compat.ir` for TileLang metadata-aware renewal, checking,
structural comparison, serialization, and exact cast services. Use
`tilelang._compat.sym.Analyzer` for the extended symbolic queries. Direct native
API calls retain the upstream behavior; importing TileLang does not silently
give every native traversal or simplifier TileLang semantics.

For example, flattening a pointer operand and its `tl.access_region` must keep
them attached to the same renewed tensor. TileLang's metadata-aware visitors and
passes maintain that relationship. Native generic traversal remains unchanged.
Query-local integer-cast normalization is not repeatedly materialized in the
output IR by simplification.

The native shared-address conversion retains its conservative Opaque effect.
ROCm helper imports do not replace the native `tvm_callback_rocm_*` functions.

## Source build

Prerequisites: Python 3.10+, C++20, CMake, Ninja, the matching installed TVM FFI
package, and SDKs for enabled devices. LLVM and Z3 are optional. Enable LLVM for
LLVM-host execution tests.

Use a clean checkout at the exact pin, including clean initialized submodules.
CMake and the provisioning helper reject a different HEAD, working-tree changes,
untracked source files, or modified submodules. Keep build directories outside
the dependency source tree.

From a workspace containing `tilelang/` and a clean dependency checkout:

```bash
cmake -S tilelang -B tilelang/build -G Ninja \
  -DTVM_TIRX_SOURCE=/path/to/clean/tvm-tirx \
  -DPython_EXECUTABLE=/path/to/venv/bin/python \
  -DPython3_EXECUTABLE=/path/to/venv/bin/python \
  -DUSE_LLVM=/path/to/llvm-config \
  -DUSE_CUDA=/usr/local/cuda -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DUSE_Z3=ON \
  -DUSE_ROCM=OFF -DUSE_ASCEND=OFF
cmake --build tilelang/build \
  --target tilelang tilelang_tirx_test_helpers --parallel
```

For this workspace's zero-patch validation, the dependency is
`dependencies/tvm-tirx-upstream-9af5544/`, a detached clean worktree. The original
sibling `tvm-tirx/` retains its historical integration HEAD even though its source
content has been reverted to the baseline; it is not the clean build input.
The PR publishes only the TileLang changes; it does not update a TIRx branch.
The acceptance wheels were built before this publication commit. Their filenames
retain the earlier TileLang HEAD; the recorded SHA256 values identify the tested
artifacts rather than the filename's Git suffix.

## Independent dependency and TileLang wheels

From `tilelang/`, with the intended virtual environment selected:

```bash
python testing/tools/prepare_tirx.py \
  --source /path/to/clean/tvm-tirx --python /path/to/venv/bin/python \
  --output-dir /path/to/wheels --build-dir /path/to/tirx-build \
  -DTVM_TIRX_USE_CUDA=/usr/local/cuda \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DTVM_TIRX_USE_LLVM=/path/to/llvm-config \
  -DTVM_TIRX_USE_ROCM=OFF

TILELANG_USE_INSTALLED_TIRX=ON uv build --wheel --no-build-isolation \
  --python /path/to/venv/bin/python --out-dir /path/to/wheels \
  -C cmake.define.TILELANG_USE_INSTALLED_TIRX=ON \
  -C cmake.define.TVM_TIRX_SOURCE= \
  -C cmake.define.USE_CUDA=/usr/local/cuda \
  -C cmake.define.USE_LLVM=/path/to/llvm-config \
  -C cmake.define.USE_ROCM=OFF
```

The helper builds and installs the pinned native dependency before the TileLang
wheel is built. `--repository` fetches the exact commit from an explicitly
selected repository; `--wheel` accepts an existing compatible dependency wheel.
The helper does not silently substitute a stock binary lacking the required
exports. Enable the native runtime sidecar for each real device backend required
by the TileLang build. SDK stubs provide source/codegen coverage, not device
execution support.

Use `--repair` or `testing/tools/repair_tirx_wheel.py` for distributable wheel
repair. Do not copy dependency libraries into the TileLang wheel to satisfy its
runtime linkage. Rebuild the TileLang wheel after native or Python adapter edits;
a successful old wheel test is not evidence about new source changes.

## Validation

Source checks should use the same Python interpreter, clean dependency source,
and freshly built libraries. For a default `tilelang/build` source build:

```bash
PYTHONPATH=/path/to/clean/tvm-tirx/python:$PWD/tilelang \
TVM_TIRX_SOURCE=/path/to/clean/tvm-tirx \
TVM_TIRX_LIBRARY_PATH=$PWD/tilelang/build/lib \
TILELANG_TIRX_TEST_LIBRARY=$PWD/tilelang/build/lib/libtilelang.so \
TILELANG_TIRX_TEST_HELPERS_LIBRARY=$PWD/tilelang/build/lib/libtilelang_tirx_test_helpers.so \
python -m pytest -q --import-mode=importlib \
  tilelang/testing/python/tirx tilelang/testing/python/tools/test_tirx_native_boundary.py
```

Run `testing/python/tools/test_tirx_packaging.py`,
`test_tirx_extension_exports.py`, `test_tirx_extension_headers.py`, and
`test_tirx_pass_boundary.py` for build and ownership boundaries. The native
boundary test snapshots native FFI function identities and Python types before
importing TileLang and verifies that they are unchanged afterwards.

Execution tests for TileLang extensions use the `compile_tilelang` fixture,
which invokes `tilelang.engine.lower` and the real TileLang pipeline. Merely
replacing a native compiler's last builder is not equivalent to this check.

Install fresh wheels into a separate environment, then run without source or
library overrides:

```bash
env -u PYTHONPATH -u TVM_TIRX_LIBRARY_PATH \
  -u TILELANG_TIRX_TEST_LIBRARY -u TILELANG_TIRX_TEST_HELPERS_LIBRARY \
  /path/to/isolated-venv/bin/python -I \
  tilelang/testing/tools/check_tirx_wheel_install.py --wheels /path/to/wheels
```

The checker verifies package, loaded compiler/runtime, template, and CUTLASS
header paths. It checks wheel contents, native FFI/type identities, and strong
implementation-symbol ownership, then runs frontend/cache, query, metadata,
and CUDA regressions outside both source trees. `--extra-site-packages` may
append a shared dependency directory without executing its editable-install
`.pth` files; TileLang, TIRx, FFI, and kernel headers must still resolve inside
the isolated environment. Test-helper libraries are not used by this check.

### Current acceptance boundaries

Acceptance recorded on October 27, 2026 (Asia/Shanghai):

- Migration suite plus `testing/python/tools/test_tirx_*.py`, `-m 'not gpu'`:
  **2512 passed**, 490 deselected, two existing deprecation warnings.
- Migration suite, `-m gpu`: **488 passed, 2 skipped**, 2435 deselected. Skips
  are float32 cases for NaN-propagating helpers with fp16/bf16-only signatures.
- After removing old test-only source include paths, the affected GPU subset
  passes again: **77 passed, 2 skipped**; this is not an additional suite count.
- Fresh repaired-wheel installation: **161 passed, 3 skipped**. The isolated
  environment lacks the optional CUTLASS Python DSL for three CuTeDSL cache tests.
- Native FFI/type identities remain unchanged; strong implementation-symbol
  overlap is zero. Installed package files match the repaired wheels byte-for-byte.

The `not gpu` selector is a test marker, not a claim of CPU-only execution.
These counts cover the migration suite and boundary checks, not every historical
TileLang test. The tested runtime reports an NVIDIA B300 (SM 10.3), with
Python 3.12, LLVM 18, and CUDA 13. The source build also covers HIP/Ascend compiler
sources using SDK stubs, not device execution. Metal/WebGPU have no device
execution qualification on this host. Hopper-specific im2col TMA is not certified
by the B300 SIMT fallback test. The local repaired wheels require glibc 2.39+;
they are not universal-platform release artifacts.

The former monorepo APIs (`te`, `topi`, Relay, the legacy scheduling stack, and
a global `tvm` namespace) are intentionally not provided. This migration is a
TileLang compiler integration with standalone TIRx, not a reimplementation of
all historical TVM APIs.
