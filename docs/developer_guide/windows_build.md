# Native builds on Windows

Windows and Linux share backend selection, source/object targets, output and
installation rules, and the Python build backend. GPU SDKs are optional; see
[Installation](../get_started/Installation.md) for setup and commands.

## Host toolchain

Ninja builds use `tilelang/_host_toolchain.py` before `project()` to discover
Visual Studio and its Windows SDK. The same helper restores the captured
compiler environment for subsequent compile/link commands and serves JIT.
Explicit compilers, `CC`/`CXX`, toolchain files, and developer shells retain
control. User launchers and compiler caches compose with environment restoration.
Visual Studio generators manage their own environment.

Native sources require clang-cl; host JIT also supports cl.exe. CUDA and HIP
share Python-environment discovery while retaining their SDK-specific setup.
CPU JIT uses `tilelang.contrib.cc.create_shared` on every platform. GNU and
MSVC compilers share process execution, ccache, timeouts, and error reporting;
MSVC additionally adapts flags, import libraries, and exported symbols.

## Windows linking

Windows embeds `tilelang_objs` in `tvm_compiler.dll`: TileLang uses TVM compiler
internals without a complete DLL interface, and exporting all internals would
exceed the PE export limit. POSIX links those objects into `libtilelang`.
Single- and multi-config generators use the same flat `build/lib` directory.

## Validation

The Windows CPU workflow builds an isolated wheel without a GPU SDK, checks
host setup, and runs the installed wheel and its cache in separate processes.
Run the wheel smoke check outside the source checkout:

```powershell
$env:TILELANG_CACHE_DIR = Join-Path $env:TEMP "tilelang-wheel-cpu-cache"
python C:\path\to\tilelang\maint\scripts\smoke_cpu_jit.py
python C:\path\to\tilelang\maint\scripts\smoke_cpu_jit.py --cache-reload
```
