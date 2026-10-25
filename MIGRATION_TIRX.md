# Standalone TIRx migration

## Status and scope

The CPU/CUDA, ROCm HIP, and Ascend CCE/PTO compiler sources now build and link
together against the standalone `tvm_tirx` compiler. Normal `import tilelang`,
the TileLang frontend, native lowering, and the TVM FFI, Cython, and NVRTC
execution adapters are operational. The `tilelang_tirx_test_passes` subset is
no longer the only buildable target.

**The standalone TIRx code migration and locally executable acceptance checks
are complete.** Original regression suites, targeted final native regressions,
and rebuilt paired wheels have been validated. CPU/CUDA and Vulkan WebGPU
numerical execution are covered, including isolated installed-wheel execution.
The all-backend source build uses HIP and Ascend SDK stubs, so those backends
have source/lowering coverage rather than device numerical validation. Metal
has code-generation coverage but no device execution on this Linux host.
This is not a claim of a public release, all-hardware qualification, or a
failure-free unmodified upstream baseline; the boundaries are recorded below.

## Dependency contract

- TileLang's starting revision is `6db1fc7` (0.1.15).
- The standalone upstream baseline is
  `9af5544df8553c35b6a85927cd2589f0a596f2fc` (0.1.0a2).
- `TVM_TIRX_COMMIT` pins the published paired integration snapshot
  `eb660c1f027189fa22cee25deeba178f534c8445`, available in
  `SiriusNEO/tvm-tirx` on branch `tilelang-standalone-integration`.
- The paired modifications to the workspace's `tvm-tirx` checkout are required.
  `TVM_TIRX_VERSION` pins `0.1.0a2+tilelang.1`; an unpatched published 0.1.0a2
  wheel is not sufficient and is never a provisioning fallback.
- `3rdparty/tvm` is removed, including its Git submodule entry. TileLang's
  remaining third-party source dependencies are CUTLASS and HIP headers.
- Python uses `apache-tvm-tirx` and `apache-tvm-ffi`. TileLang does not bundle
  copies of their compiler, runtime, or FFI shared libraries in its wheel.
- `tilelang.tvm` is a TileLang-scoped adapter. It does not install a global
  `tvm` module alias or make the former TVM monorepo an interchangeable dependency.

The dependency changes include native lexical scopes, exported data-type and
operator-signature APIs, expression-bearing attributes with binding-aware
traversal, dialect-call vectorization, packed storage ABI validation, exact
`PrimType` serialization, explicit-step loop code generation, and CUDA vector
and host-codegen support. CMake checks the essential extension headers in both
source and installed-dependency configurations.

## Source build

Prerequisites are Python 3.10+, C++20, CMake, Ninja, a matching installed TVM FFI
package, and the SDKs for enabled devices. LLVM and Z3 are optional build
features; enable LLVM for LLVM-host numerical tests.

From a workspace containing sibling `tilelang` and `tvm-tirx` directories:

```bash
cmake -S tilelang -B tilelang/build -G Ninja \
  -DTVM_TIRX_SOURCE="$PWD/tvm-tirx" \
  -DPython_EXECUTABLE=/path/to/venv/bin/python \
  -DPython3_EXECUTABLE=/path/to/venv/bin/python \
  -DUSE_LLVM=/path/to/llvm-config \
  -DUSE_CUDA=/usr/local/cuda -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DUSE_Z3=ON \
  -DUSE_ROCM=OFF -DUSE_ASCEND=OFF
cmake --build tilelang/build \
  --target tilelang tilelang_tirx_test_helpers --parallel
```

`../tvm-tirx` is the default source dependency location. CPU-only builds can use
`-DUSE_CUDA=OFF`. Keep the selected interpreter free of editable installs of
another TileLang checkout or the former TVM fork.

The `tilelang` target also builds the enabled standalone runtime sidecars,
including `libtvm_runtime_cuda` and `libtvm_runtime_rocm`. Linking only the core
runtime is insufficient: its device APIs are registered by those sidecars.

## Installed dependency and wheels

The supported provisioning entry point, run from `tilelang/` with the desired
virtual environment activated, is:

```bash
python testing/tools/prepare_tirx.py --source ../tvm-tirx \
  -DTVM_TIRX_USE_CUDA=/usr/local/cuda \
  -DTVM_TIRX_USE_LLVM=/path/to/llvm-config
export TILELANG_USE_INSTALLED_TIRX=ON
export UV_FIND_LINKS="$PWD/dist/tirx"
export PIP_FIND_LINKS="$PWD/dist/tirx"
USE_CUDA=ON USE_ROCM=OFF USE_ASCEND=OFF \
  uv pip install --no-build-isolation -e .
```

For a clean checkout without the sibling source directory, fetch the published
paired snapshot through the same helper:

```bash
python testing/tools/prepare_tirx.py \
  --repository https://github.com/SiriusNEO/tvm-tirx.git \
  -DTVM_TIRX_USE_CUDA=/usr/local/cuda \
  -DTVM_TIRX_USE_LLVM=/path/to/llvm-config
```

The helper fetches the exact `TVM_TIRX_COMMIT`, not the branch tip. CI can select
this repository with `TVM_TIRX_REPOSITORY`; this draft does not change shared
repository variables or publish a dependency release.

Use `--wheel <path-or-URL>` instead of `--source` for an existing paired wheel.
The helper validates its distribution/version, installs build prerequisites,
builds and installs the dependency independently, and checks its CMake package
and requested runtime sidecars. `--repair` invokes the platform wheel-repair
tool while enforcing separate ownership of TIRx and FFI libraries.

First build/install the patched standalone dependency, with the needed
`TVM_TIRX_USE_LLVM`, `TVM_TIRX_USE_CUDA`, and `TVM_TIRX_USE_Z3` settings. The
paired build currently uses `TVM_TIRX_EXPORT_ALL_SYMBOLS=ON` for extension linkage.
Then configure TileLang with:

```bash
-DTILELANG_USE_INSTALLED_TIRX=ON \
-Dtvm_tirx_DIR=/path/to/prefix/lib/cmake/tvm_tirx
```

The CMake package can also be discovered through the selected interpreter's
`tvm_tirx.config --cmakedir`. This configuration links the independent package;
it does not build a second dependency under TileLang.

Use separate build directories for source and installed-dependency builds.
`uv build --wheel --no-build-isolation --python /path/to/venv/bin/python` accepts
these switches as `-C cmake.define.NAME=VALUE`. Build the standalone wheel first,
install it into the isolated build environment, and only then build TileLang's
wheel with `TILELANG_USE_INSTALLED_TIRX=ON`.

An installed-wheel check must run outside both source trees, with `python -I`
and without `PYTHONPATH` or source-library overrides. Check that importing
TileLang does not put `tvm` in `sys.modules`, and inspect the wheel for duplicate
TIRx/FFI libraries as well as running numerical kernels. Existing wheels must be
rebuilt whenever either checkout's native implementation changes.

### Validated local artifacts

The paired, repaired wheels are in `dist/tirx-rebase-validated/` relative to the
parent workspace, alongside `SHA256SUMS`:

```text
apache_tvm_tirx-0.1.0a2+tilelang.1-py3-none-manylinux_2_39_x86_64.whl
tilelang-0.1.15+cuda.git6db1fc77-cp39-abi3-manylinux_2_39_x86_64.whl
```

Both contain the final native changes. Their filenames retain the validation
checkout's pre-commit version metadata; publishing the review commits changes
the source pin, documentation, and build-directory ignore rules, not the native
implementation. TileLang's wheel was built with
`TILELANG_USE_INSTALLED_TIRX=ON` and an empty `TVM_TIRX_SOURCE`, against the
separately installed dependency. Its build has all SDK-stub options disabled;
CUDA uses the real SDK, while ROCm and Ascend are disabled. These are Linux
x86-64 CPU/CUDA artifacts, not all-device distribution wheels. Their actual
repair tag is `manylinux_2_39`, not a promise of compatibility with older
manylinux hosts. The dependency's local version identifies this patched pair,
not an unmodified upstream release.

The installed libraries use relative `$ORIGIN` dependency paths. Validation
checks that TileLang, TIRx, FFI, and their loaded compiler/runtime libraries
come from the isolated environment rather than either source tree. Torch and
other test utilities may be shared through an explicit site-packages path;
that path does not execute the shared environment's editable-install hooks.
CuTeDSL 4.7.0, including `nvidia-cutlass-dsl-libs-core==4.7.0`, and
`wgpu==0.32.0` are installed in the isolated environment for optional cache and
WebGPU numerical coverage. Installed Python and native files are also checked
against the corresponding wheel contents, not only their installation paths.

## CI and build entry points

- `.github/actions/setup-tirx` provisions the dependency before project or test
  requirements are installed. Configure `TVM_TIRX_WHEEL`, or
  `TVM_TIRX_REPOSITORY` together with the matching committed source pin.
  `TVM_TIRX_REVISION` cannot override `TVM_TIRX_COMMIT` without updating the file.
- Wheel URLs may contain `{platform}` and `{machine}` placeholders. Source and
  wheel paths must be accessible inside the actual build environment, including
  the manylinux container when using cibuildwheel.
- `TVM_TIRX_CMAKE_DEFINES` accepts a JSON object of dependency CMake settings.
  Source builds need the SDKs for every enabled dependency runtime. Linux
  distribution builds retain CUDA, ROCm, and Ascend support; the CUDA toolkit
  installed by the container hook alone is not sufficient to build the ROCm
  sidecar. Supply its SDK or a matching prebuilt paired dependency.
- Dependency wheels are repaired and uploaded separately, then installed through
  a local wheel search path by GPU wheel-test jobs. TileLang wheel repair rejects
  bundled TIRx or FFI shared libraries.
- `maint/scripts/run_perf_regression.sh` provisions each migrated checkout in
  its own environment and includes the dependency wheel hash in its wheel cache
  identity. Comparisons against an unmodified pre-migration baseline retain
  that baseline's original build process.
- Docker builds use the local TileLang checkout and the same provisioning tool;
  see `docker/README.md` for the new build-context and dependency arguments.
- The current local dependency version is for the paired workspace/artifacts,
  not a claim of an upstream release. Public release requires publishing a
  public paired dependency first, updating all version/source pins, and setting
  `TVM_TIRX_RELEASE_READY=true`. The release job rejects a local dependency
  version and checks that the selected public version exists before upload.

## Native compiler contracts

- Tensors, tensor types, layouts, regions, and typed pointers are native TIRx
  objects. A TileLang tensor view does not reintroduce the old Buffer IR.
- Tensor geometry, storage offsets, swizzles, and alignment must remain attached
  to the native tensor/layout boundary. Storage reuse also updates retained
  access regions; changing only the pointer is insufficient.
- `tirx.scope` is lexical allocation/binding scope, not `tirx.compute_scope`.
  Warp-specialized concurrency is represented explicitly by `tl.ws_concurrent`.
- Native function, loop, and call attributes may contain expressions referring
  to bound variables, including body-local definitions. Traversal, substitution,
  renaming, equality, and JSON round trips preserve those bindings.
- `CompiledArtifact.host_mod` remains an IR module. The executable runtime module
  is `rt_mod`; native `Executable` owns JIT/export operations, not `tvm_ffi.Module`.
- CUDA host adapters preserve explicit streams, typed callback pointers, aligned
  tensor-map storage, and descriptor lifetimes. Raw C-ABI hosts resolve global
  packed callbacks without assuming a TVM FFI module context has been installed.
- Cython compiles the already-postprocessed device source rather than invoking
  source postprocessing a second time. Cache hits retain that processed source.
- Native CUDA emits function declarations and definitions in a deterministic
  global-name order. Module-wide IKET event IDs therefore do not depend on the
  iteration order of the native IR module's function map.
- Metal SIMD-group matrices and Metal 4 MPP cooperative tensors are emitted by
  the standalone backend. Cooperative accumulators use independent named MPP
  objects, preserve distinct source/destination accumulators, and retain the
  fragment layout across load, multiply, and store. MLX swizzles distinguish
  physical and logical block indices instead of applying the mapping twice.
- WebGPU merged byte arenas preserve nested typed-view offsets. Packed shared
  storage uses atomic 32-bit words, including subword updates, and both shared
  scopes emit workgroup barriers. Half conversion uses WGSL pack/unpack
  operations; signed packed `int8x4` remains a 32-bit value for packed dot
  products. Integer index normalization preserves vector lane counts.

### Packed storage and cache identity

`tirx.packed_storage_dtypes` describes the physical tensor storage accepted by
the packed ABI. Shapes, strides, offsets, storage types, and device constraints
are validated. Packed FP4 uses packed-byte storage; unpacked FP4 uses one byte
per logical element. These are different DLPack encodings even when a dtype's
string representation is identical.

`PrimType` JSON and script serialization preserve code, bit width, and lane
count. Kernel-parameter JSON stores native `PrimType` objects, preserving that
encoding and shared symbolic shape-variable identity across parameters. Old
string-based parameter records remain readable. This guarantee does not imply
that every unrelated raw `DLDataType` field in the dependency has the same
custom serialization hooks.

Both kernel and device-binary caches fingerprint the TileLang, loaded TIRx
compiler/runtime, and FFI binaries when library stamps are enabled. Export after
a disk-cache hit copies the cached executable instead of treating a loaded FFI
module as an exportable compiler module.

### Intentionally retired APIs

The former TE/Schedule-based Carver policies and templates are not implemented
by standalone TIRx. They are retired explicitly, not silently replaced by an
empty schedule or a no-op pass. Native TileLang autotuning, hardware descriptions,
and hints remain available. See `docs/tutorials/auto_tuning.md`.

## Validation

Source-tree regression commands, from the parent workspace:

```bash
export PYTHONPATH="$PWD/tilelang:$PWD/tvm-tirx/python"
export TVM_TIRX_LIBRARY_PATH="$PWD/tilelang/build/lib"
export TILELANG_TIRX_TEST_LIBRARY="$PWD/tilelang/build/lib/libtilelang.so"
export PATH="/usr/local/cuda/bin:$PATH"
TILELANG_DISABLE_CACHE=1 python -m pytest -q \
  tilelang/testing/python/tirx/test_tirx_frontend.py \
  tilelang/testing/python/tirx/test_tirx_cuda_host_services.py
TILELANG_DISABLE_CACHE=0 python -m pytest -q tilelang/testing/python/cache
TILELANG_LIGHT_IMPORT=1 TILELANG_DISABLE_CACHE=1 python -m pytest -q \
  tilelang/testing/python/tirx \
  --ignore=tilelang/testing/python/tirx/test_tirx_frontend.py
```

Light-import mode intentionally excludes tests that require the normal frontend;
run those separately without `TILELANG_LIGHT_IMPORT`. Native dependency tests
cover the changed IR, symbolic analysis, host ABI, CUDA code generation, and
vectorizer. Run the original TileLang suites as well: a collection-only pass is
not execution coverage, and source generation is not numerical validation.

CUDA runtime kernels must target the actual execution architecture. In
particular, CuTe cubins compiled for Hopper cannot be used as a numerical test
on a Blackwell GPU. Architecture-specific TCGEN05 tests need the appropriate
`a` target suffix. CUDA 13 does not provide an SM70 compiler target.

After installing both wheels in a separate environment, run
`python -I testing/tools/check_tirx_wheel_install.py --wheels <wheel-directory>`
using that environment's interpreter. Unset `PYTHONPATH`,
`TVM_TIRX_LIBRARY_PATH`, and the two `TILELANG_TIRX_TEST_*_LIBRARY` overrides.
The checker verifies installed package and loaded compiler-library paths,
rejects vendored TVM or duplicated dependency libraries, and executes the
frontend CPU/CUDA numerical and disk-cache/export suites. Optional
`--extra-site-packages <directory>` adds a preinstalled Torch/test dependency
directory without executing its editable-install `.pth` files.

## Acceptance outcome and boundaries

- Source builds and the installed-dependency build complete without the former
  TVM checkout. The dependency is independently packaged, and TileLang's wheel
  contains no second TIRx or FFI compiler/runtime implementation.
- Original CUDA/runtime/debug regressions and targeted final native
  storage/codegen regressions have been rerun. Migration-specific failures in
  the executed suites are resolved; independent baseline failures remain
  explicitly recorded rather than disabled or counted as successes.
- Both wheels have been rebuilt after the final native changes. Isolated
  CPU/CUDA frontend, cache/export, compile-only CLI, and WebGPU numerical checks
  pass without either source tree on the compiler import/library path.
- HIP, Ascend, and Metal still require numerical qualification on matching
  hardware and real SDKs. Architecture-specific skips and source-generation
  checks do not certify execution or performance parity on those devices.
- These paired local artifacts are not a public release. Public dependency
  publication and release-pin updates follow the release contract above;
  unpatched upstream TIRx wheels are not substitutes for this dependency.

### Final validation results

Acceptance combines original-suite runs with targeted reruns after the last
native changes and isolated installation of the rebuilt pair. CUDA tests run
on compute capability 10.3 with CUDA 13.1; LLVM 18 and Z3 are enabled. HIP and
Ascend use SDK stubs and have source/lowering coverage only.

| Suite | Result |
| --- | --- |
| Native TIRx integration, original transforms, CPU and LLVM | 3,214 passed, 9 skipped |
| Complete original and native Ascend scheduling/lowering | 1,429 passed, 391 skipped |
| Original issue regressions | 317 passed, 21 skipped |
| Original language regressions, four shards | 2,057 passed, 40 skipped |
| Original runtime, JIT, kernel, CUDA, debug and Metal suites | 1,012 passed, 49 skipped |
| Profiler, including available CUPTI/performance paths | 103 passed, 9 skipped |
| Original cache suite, cache enabled and isolated directory | 72 passed |
| Autotuner, cache enabled and isolated directory | 45 passed, 1 skipped |
| Normal default-build frontend import and CPU/CUDA execution | 49 passed |
| WebGPU native source and Vulkan numerical tests | 113 passed |
| TileLang WebGPU source and FP16/FP32 numerical GEMM | 7 passed |
| Standalone CUDA packed arithmetic, including fused FMA | 42 passed |
| Dependency provisioning and wheel-boundary unit tests | 34 passed |
| Sparse compression and dense-reference numerical comparisons | 6 passed; compute-sanitizer reports 0 errors |
| Final native CUDA ordering, Metal, WebGPU and index-narrowing regressions | 173 passed |
| Final isolated-wheel CPU/CUDA frontend and complete cache/export suites | 121 passed |
| Installed-wheel tools, including isolated compile-only CLI subprocesses | 80 passed, 1 skipped |
| Installed-wheel Metal source and WebGPU numerical regressions | 29 passed, 23 Metal-device tests skipped |

Counts describe individual invocations and are not additive across overlapping
suites. The final installed-backend run executes all six FP16/FP32 WebGPU
numerical GEMMs; its remaining 23 skips require Metal hardware/runtime support.
The final isolated checker also executes the three optional CuTeDSL cache tests
without skips. Local final logs are `/tmp/tirx-wheel-check443.log`,
`/tmp/tirx-installed-tools440.log`, `/tmp/tirx-installed-backends441.log`, and
`/tmp/tirx-native440.log`.

The auxiliary source-only sweep initially reported 11 compile-only CLI import
failures because those tests deliberately invoke `python -I`, while its source
environment does not install TileLang. They pass in the installed-wheel tools
suite without removing isolation or injecting either source tree. Its remaining
IKET event-order failure is fixed by deterministic native CUDA function
ordering; the original strict checks pass for both forward and reverse module
insertion order.

Independent TIRx regression triage compares the same tests against an unmodified
`9af5544df8553c35b6a85927cd2589f0a596f2fc` CUDA/LLVM build with its exact FFI pin.
Of 38 failures in the initial dependency run, 34 also fail on that baseline.
The four migration-specific failures were stale packed-API expectations for
empty-tensor stride checks; the complete packed-API file subsequently passes
14 tests. Baseline failures concern required launch-size assembly directives,
IKET startup timeout, native tile-load offsets, architecture-specific cubins,
TMA offset fields, TCGEN05 int8 target support, TMEM dispatch, and native local
allocation views. They are not disabled or represented as migration successes.

The WGMMA runtime scale-predicate test additionally reproduces an NVRTC crash
on identical baseline/rebased CUDA source. Compiling that source with nvcc also
reports a ptxas internal compiler error. This toolkit failure is excluded from
the dependency sweep and remains an explicit validation boundary.

A later standalone symbolic/IR/CUDA/backend sweep passes 2,270 tests, skips
17, and reports 8 expected failures. Its one additional failure,
`test_e2m1_dequantize`, also reproduces on the unmodified TIRx baseline with
the same FFI revision: vector `uint8x4` to FP4 reinterpretation rejects a
four-byte source for a two-byte destination. The baseline test and production
check are not weakened to hide that pre-existing failure.

The original WebGPU source-generation smoke test is not a numerical reference.
The unmodified TileLang baseline emits the same `<=` fragment-ownership
predicate for its serial `T.grid` GEMM, as well as unsupported `u8` workgroup
storage. It is retained as an original source-generation regression, not
counted as numerical acceptance. The separate WebGPU runtime regression uses
explicit `T.Parallel` output ownership and checks FP16/FP32 GEMM against NumPy,
including partial tiles and multiple reduction tiles. Native byte-arena tests
also execute vector, scalar, signed-subword, and reinterpreted accesses through
Vulkan WebGPU. The optional `wgpu` Python package is a test dependency, not a
TileLang runtime dependency.
