# Native builds on Windows

Windows uses the same backend options, source lists, object targets, and Python
build backend as Linux. Choose CPU, CUDA, or HIP through `USE_CUDA` and
`USE_ROCM`; GPU SDK packages belong to the optional `nvcc` and `rocm` extras.
See [Installation](../get_started/Installation.md) for SDK setup and commands.

## Host toolchain and device SDKs

`cmake/HostToolchain.cmake` runs before `project()`. On Windows with Ninja it
uses the stdlib-only `tilelang/_host_toolchain.py`, which also serves JIT
compilation, to discover Visual Studio and its Windows SDK. A developer shell,
explicit compilers, `CC`/`CXX`, or a toolchain file retain control over compiler
selection. Wheel builds default to Ninja unless `CMAKE_GENERATOR` is explicitly
set. Visual Studio generators manage their own environment; native TileLang
sources currently require their ClangCL toolset, since cl.exe cannot compile
the existing GCC/Clang builtins and some TVM declarations. Host JIT compilation
continues to support cl.exe as well as clang-cl.

Configuration captures only compiler-related environment variables. Compile
and link launchers restore them for each subprocess, including a later
`cmake --build` from an ordinary shell. User launchers and compiler caches
compose with that environment launcher. No SDK include or library flags are
appended to cached global flags.

CUDA discovery runs after `project()`, independently of the host setup. An
explicit `USE_CUDA=OFF` bypasses CUDA discovery, including pip SDK synthesis.
HIP SDK discovery remains owned by the ROCm backend. Generic PEP 517 build
dependencies contain neither SDK; isolated CUDA release builds request SDK
packages through scikit-build-core's `build.requires` setting.

## Shared targets and Windows ABI constraints

Backend CMake files collect sources into `tilelang_objs` on every platform.
POSIX builds link those objects into `libtilelang`; Windows embeds them in
`tvm_compiler.dll` because TileLang calls TVM compiler internals that are not
exported through a complete Windows DLL interface. Exporting all TVM internals
would also exceed the PE DLL export limit. This small final-link adaptation
preserves the common source and object graph.

Native libraries and Python extensions share flat `build/lib` output paths for
single- and multi-config generators, as well as installation rules. Windows DLL
import libraries generated for SDK packages are invalidated by DLL content,
so updating a dependency cannot reuse an import library for old exports.

CPU JIT delegates to `tilelang.contrib.cc.create_shared` on every platform.
The Windows compiler adapter selects MSVC flags, SDK environment, DLL suffix,
and exports; POSIX retains its existing compiler path. Generated CPU wrappers
export the same `call` entry point as other source adapters. The C target uses
the Cython execution backend; LLVM execution is a separate backend.

## Validation

The Windows CPU build workflow builds an isolated wheel without a GPU SDK.
It checks explicit compiler selection, separate configure/build environments,
and repeated configuration, then executes the installed wheel and loads its
CPU kernel cache in a fresh process with compilation disabled.

Run the same wheel smoke check outside the source checkout:

```powershell
$env:TILELANG_CACHE_DIR = Join-Path $env:TEMP "tilelang-wheel-cpu-cache"
python C:\path\to\tilelang\maint\scripts\smoke_cpu_jit.py
python C:\path\to\tilelang\maint\scripts\smoke_cpu_jit.py --cache-reload
```

For a development checkout, run:

```powershell
python -m pytest testing/python/contrib/test_tilelang_host_toolchain.py `
  testing/python/contrib/test_tilelang_contrib_cc.py -q
```

GPU execution still requires the selected runtime and a supported physical GPU.
Building a CPU wheel or generating device source is not GPU execution coverage.
