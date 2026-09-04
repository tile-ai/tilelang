# Compile-Only Tool

`python -m tilelang.tools.compile_only` lowers a TileLang program to
inspectable kernel source without running it. No GPU, CUDA driver, or device
compilation is required: the tool stops after
`tilelang.lower(..., enable_device_compile=False)` and writes the generated
kernel source to a file. It backs the TileLang integration in
[Compiler Explorer](https://godbolt.org) and is equally useful for inspecting
locally what `lower()` produces for a kernel.

## Requirements

The tool ships with the standard TileLang installation and has no optional
dependencies. The default `c` target works on any machine, including CPU-only
hosts. `--target cuda` requires a wheel whose CUDA codegen FFI is compiled in:
Linux wheels include it and can emit CUDA source without a GPU present, while
macOS Metal wheels do not and fail with a clear
`CUDA codegen FFI missing` message instead of an `AttributeError`.

## Quick Start

Write a self-contained example that defines a kernel but never launches it:

```python
# example.py
import tilelang
import tilelang.language as T


@tilelang.jit
def add(A, B):
    N = 64
    A: T.Tensor((N,), T.float16)
    B: T.Tensor((N,), T.float16)
    C = T.empty((N,), T.float16)
    with T.Kernel(1, threads=64):
        for i in T.Parallel(N):
            C[i] = A[i] + B[i]
    return C
```

Compile it to CPU C source:

```bash
python -I -m tilelang.tools.compile_only --output_file out.c example.py
```

`out.c` then contains the generated kernel source:

```c
// tilelang target: {"kind":"c","tag":"","keys":["cpu"],"host":{"kind":"c","tag":"","keys":["cpu"]}}
#include <tl_templates/cpp/common.h>

#ifdef __cplusplus
extern "C"
#endif
int32_t add_kernel(half* A, half* B, half* C);
#ifdef __cplusplus
extern "C"
#endif
int32_t add_kernel(half* A, half* B, half* C) {
  for (int32_t i = 0; i < 8; ++i) {
    *(half8*)(C + (i * 8)) = (*(half8*)(A + (i * 8)) + *(half8*)(B + (i * 8)));
  }
  return 0;
}
```

CUDA source generation works the same way and still needs no GPU; the
architecture is pinned instead of detected:

```bash
python -I -m tilelang.tools.compile_only --target cuda --output_file out.cu example.py
```

```cuda
extern "C" __global__ void __launch_bounds__(64, 1) add_kernel(const half_t* __restrict__ A, const half_t* __restrict__ B, half_t* __restrict__ C) {
  C[((int)threadIdx.x)] = (A[((int)threadIdx.x)] + B[((int)threadIdx.x)]);
}
```

## How the Input Is Processed

- The input file is imported as a regular Python module, so its top-level code
  runs. Keep examples compile-only: define kernels, do not launch them or
  allocate tensors at import time. Running the CLI with `python -I` (isolated
  mode) is recommended, and is how Compiler Explorer invokes it.
- The first `@tilelang.jit` function or `PrimFunc` in module order is selected.
  Objects are inspected in definition order, so a `PrimFunc` defined before a
  `@tilelang.jit` kernel wins.
- The kernel is lowered but never device-compiled or executed, so no tensor
  arguments are needed and no GPU is touched.

## CLI Reference

```text
python -I -m tilelang.tools.compile_only [--target TARGET] --output_file OUTPUT input_file
```

| Argument | Meaning |
| --- | --- |
| `input_file` | Compile-only example file. Imported, never launched. |
| `--output_file` | Required destination for the generated kernel source. Must not name the input file (symlinks to it are also rejected). |
| `--target` | Explicit target. Defaults to `c`. |

On success the CLI exits with `0` and writes the source to `--output_file`. On
any failure it exits with `1`, prints a single
`tilelang compile-only error: ...` line to stderr, and leaves no stale output
behind: a previous artifact at `--output_file` is removed before compilation,
so a failed rerun cannot leave an earlier run's source for a consumer such as
Compiler Explorer to read.

## Targets

Targets must be explicit. `auto` performs device detection and is therefore
rejected, as a plain string and in JSON form:

```text
$ python -I -m tilelang.tools.compile_only --target auto --output_file out.c example.py
tilelang compile-only error: target must be explicit; do not use auto
```

| `--target` value | Behavior |
| --- | --- |
| `c` (default) | CPU C source. Works on every wheel, no GPU needed. |
| `cuda` | CUDA source pinned to `sm_80`, so no device detection happens. |
| `cuda -arch=sm_90` | CUDA source for the given architecture. Options must look like `-key=value`. |
| `{"kind": "cuda", "arch": "sm_90"}` | JSON target. A bare `{"kind": "cuda"}` is pinned to `sm_80`; `{"kind": "auto"}` is rejected. |
| `auto` | Rejected: `target must be explicit; do not use auto`. |

On wheels without the CUDA codegen FFI (for example macOS Metal wheels), every
CUDA target form fails softly with
`tilelang compile-only error: CUDA codegen FFI missing (e.g. macOS Metal
wheel); use default --target c`.

## Error Reporting

Compile problems are reported as diagnostics on stderr with exit code `1`, not
as tracebacks about missing GPUs or drivers:

```text
$ python -I -m tilelang.tools.compile_only --output_file out.c broken.py
tilelang compile-only error: no @tilelang.jit kernel or PrimFunc found
```

Syntax errors in the input, missing input files, and empty kernels are
reported the same way.

## Line Directives

When the installed build registers the `tl.emit_line_directives` pass config
(`PassConfigKey.TL_EMIT_LINE_DIRECTIVES`), the tool enables it automatically
and the generated source carries `#line <n> "<input_file>"` directives that map
each statement back to the originating line of the example. Compiler Explorer
uses these to link generated source to the input pane. Older wheels without
the config key simply produce source without `#line` directives; no flag is
needed either way.

## Programmatic API

The same functionality is available in Python:

```python
from tilelang.tools.compile_only import compile_kernel_source

source = compile_kernel_source(add.get_tir())              # default target "c"
cuda_source = compile_kernel_source(add.get_tir(), "cuda")  # pinned to sm_80
```

- `compile_kernel_source(func, target="c")` lowers a `PrimFunc` (for example
  from `JITImpl.get_tir()`) and returns the non-empty kernel source string.
- `resolve_target(target)` maps a CLI-style target string to the explicit
  target passed to `lower()`, applying the rules from the table above.
- `cuda_codegen_available()` reports whether the installed wheel can lower
  CUDA targets.

## Limitations

- Only the first `@tilelang.jit` kernel or `PrimFunc` in the module is
  compiled; additional kernels in the same file are ignored.
- The output is lowered kernel source, not a compiled binary: nothing is
  passed to `nvcc`, and PTX/SASS is out of scope.
- The input file is executed with the invoking Python interpreter. Treat it
  like any script you run: `python -I` isolates it from your environment but
  does not sandbox it.
