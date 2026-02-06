# Debugging Tile Language Programs

<div style="text-align: left;">
<em>Author:</em> <a href="https://github.com/LeiWang1999">Lei Wang</a>
</div>

## Overview

A Tile Language program (hereafter referred to as a *program*) is transformed into a hardware-executable file through several stages:

1. The user writes a Tile Language program.
2. The program undergoes multiple *Passes* for transformation and optimization (the *lower* stage, see `tilelang/engine/lower.py`), finally producing an intermediate representation (e.g., LLVM or C for CPU, CUDA for NVIDIA GPUs, etc.).
3. The generated code is compiled by the respective compiler (e.g., nvcc) into a hardware-executable file.

```{figure} ../_static/img/overview.png
:width: 300
:alt: Overview of the compilation process
:align: center

```

During this process, users may encounter roughly three categories of issues:

- **Generation issues**: The Tile Language program fails to generate a valid hardware-executable file (i.e., errors during the lowering process).
- **Correctness issues**: The resulting executable runs, but produces incorrect results.
- **Performance issues**: The executable runs with performance significantly below the expected theoretical hardware limits.

This tutorial focuses on the first two issues—how to debug generation and correctness problems. Performance tuning often requires using vendor-provided profiling tools (e.g., **Nsight Compute**, **rocProf**, etc.) for further hardware-level analysis, which we will address in future materials.

Below, we take matrix multiplication (GEMM) as an example to demonstrate how to write and debug a Tile Language program.

## Matrix Multiplication Example

In **Tile Language**, you can use the **Tile Library** to implement matrix multiplication. Here's a complete example:

```python
import tilelang
import tilelang.language as T

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    # ...existing code...

# 1. Define the kernel (matmul) with the desired dimensions
func = matmul(1024, 1024, 1024, 128, 128, 32)

# 2. Compile the kernel into a torch function
# ...existing code...
```

## Debugging Generation Issues

TileLang essentially performs *progressive lowering*. For example, a `T.copy` may first be expanded into `T.Parallel` (see the pass `LowerTileOP`), which is then expanded again, eventually resulting in lower-level statements that can be translated to CUDA C code.

```{figure} ../_static/img/ir_transform_diagram.png
:width: 400
:alt: IR transformation diagram
:align: center

```

When the code fails to generate (for instance, a compilation error occurs), you do **not** necessarily need to jump directly into C++ passes to debug. Instead, you can first inspect the intermediate representations (IR) in Python by printing them.

For example, consider a case where a simple `T.copy` in 1D causes the lowering process to fail. The snippet below illustrates a simplified version of the problem (based on community Issue #35):

```python
@T.prim_func
def main(Q: T.Tensor(shape_q, dtype)):
    # ...existing code...
```

The TileLang lower process might yield an error such as:

```text
File "/root/TileLang/src/target/codegen_cuda.cc", line 1257
ValueError: Check failed: lanes <= 4 (8 vs. 4) : Ramp of more than 4 lanes is not allowed.
```

This indicates that somewhere during code generation, an unsupported vectorization pattern was introduced (a ramp of 8 lanes). Before diving into the underlying C++ code, it is helpful to print the IR right before code generation. For instance:

```python
device_mod = tir.transform.Filter(is_device_call)(mod)
# ...existing code...
```

## Debugging Correctness Issues

Sometimes, the kernel compiles and runs but produces incorrect results. In such cases, there are two main strategies to help debug:

1. **Use post-processing callbacks to inspect or modify the generated CUDA code.**
2. **Use the built-in `T.print` debugging primitive to inspect values at runtime.**

### Post-Processing Callbacks for Generated Source

After code generation (in the codegen pass), TileLang calls a callback function (if registered) to allow post-processing of the generated source code. In `src/target/rt_mod_cuda.cc`:

```cpp
std::string code = cg.Finish();
if (const auto *f = Registry::Get("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).operator std::string();
}
```

Hence, by registering a Python function named `tilelang_callback_cuda_postproc`, you can intercept the final CUDA code string. For example:

```python
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.callback import register_cuda_postproc_callback

@register_cuda_postproc_callback
def tilelang_callback_cuda_postproc(code, _):
    print(code) # print the final CUDA code
    code = "// modified by tilelang_callback_cuda_postproc\n" + code
    return code

kernel = tilelang.compile(matmul, target="cuda")
kernel_source = kernel.get_kernel_source()
print(kernel_source)
'''
// modified by tilelang_callback_cuda_postproc
#include "cuda_runtime.h"
...
'''
```

### Runtime Debug Prints with `T.print`

TileLang provides a built-in debugging primitive called `T.print` for printing within kernels. Be mindful of concurrency and thread synchronization when using it in GPU code. Below are some examples showing how to print buffers, variables, and other data inside TileLang programs.

1. **Printing an Entire Buffer**

```python
def debug_print_buffer(M=16, N=16):
    # ...existing code...
```

2. **Conditional Printing**

```python
def debug_print_buffer_conditional(M=16, N=16):
    # ...existing code...
```

3. **Printing Thread Indices or Scalar Values**

```python
def debug_print_value_conditional(M=16, N=16):
    # ...existing code...
```

4. **Printing Fragment (Register File) Contents**

```python
def debug_print_register_files(M=16, N=16):
    # ...existing code...
```

5. **Adding a Message Prefix**

```python
def debug_print_msg(M=16, N=16):
    # ...existing code...
```

The output messages will include something like:

```text
msg='hello world' BlockIdx=(0, 0, 0), ThreadIdx=(0, 0, 0): 0
```

### Visual Layout Inference For TileLang
 The **Visual Layout Inference** tool automatically generates visual diagrams that illustrate the mapping between logical indices, thread IDs, and register file locations.

When TileLang performs layout inference, it determines how fragment buffers are distributed across threads. The visual layout tool captures this information and generates:
1. **Textual output**: A human-readable description of the layout mapping
2. **Visual diagrams**: Color-coded plots showing the thread-to-data mapping

The visual layout inference tool is controlled through the `TL_LAYOUT_VISUALIZATION_ENABLE` and `TL_LAYOUT_VISUALIZATION_FORMATS` pass configuration. By default, `TL_LAYOUT_VISUALIZATION_ENABLE` is **disabled** to avoid performance overhead during compilation.

When enabled, `TL_LAYOUT_VISUALIZATION_FORMATS` accepts string values to control output formats:
- "txt": Text output only (same as default)
- "all": Generates all formats (TXT, PDF, PNG, SVG)
- "png": Generate PNG format only
- "pdf": Generate PDF format only
- "svg": Generate SVG format only
- "txt,svg": Generate multiple formats (comma-separated) in addition to text output

The output messages of "txt" will include something like:
```
C_local inferenced layout:
  Shape: [32, 32] -> [8]
  Thread: _j // 16 * 64 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2
  Index:  [_j % 16 // 8 * 4 + _i % 16 // 8 * 2 + _j % 2]
```

## Conclusion

By carefully examining intermediate representations (IR) before final code generation—and by leveraging runtime printing through `T.print`—one can quickly diagnose where index calculations, copy logic, or other kernel operations deviate from the intended behavior. This two-pronged approach (inspecting IR transformations and using runtime prints) is often sufficient for resolving generation and correctness issues in TileLang programs.

For advanced performance tuning (e.g., analyzing memory bandwidth or occupancy), more specialized profiling tools such as **Nsight Compute**, **rocProf**, or vendor-specific profilers may be required. Those aspects will be covered in future documents.
