# Working with Native TIRx IR

TileLang keeps its public DSL spellings, including `import tilelang.language as T`,
`@T.prim_func`, `@tilelang.jit`, tensor annotations, and kernel scopes. The compiler
objects returned by those entry points use the native TIRx interfaces. Extension
code that inspects or rewrites IR must use these interfaces as well.

`tilelang.tvm` is an alias of `tvm_tirx`. It exposes the same object system and
compiler provider; it does not install a top-level `tvm` module or recreate the
old `PrimFunc`, `Buffer`, visitor, or registry interfaces.

## Inspect a Kernel

```python
import tilelang.language as T
import tvm_ffi
from tvm_tirx import ir, tirx


@T.prim_func
def copy(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
    for i in T.serial(8):
        B[i] = A[i]


assert isinstance(copy, tirx.Function)
assert isinstance(copy.body, ir.SeqStmt)

tensor_parameters = [p for p in copy.params if isinstance(p.ty, tirx.TensorType)]
assert len(tensor_parameters) == 2
for parameter in tensor_parameters:
    print(parameter.name, parameter.ty.dtype, list(parameter.ty.shape))

loads = []
tvm_ffi.structural_walk(copy, (ir.TensorLoad, loads.append), order="post")
assert len(loads) == 1
assert tirx.analysis.check_well_formed(copy)
assert not tirx.analysis.undefined_vars(copy)
```

`JITImpl.get_tir()` returns the same native `tirx.Function` representation. Scalar
parameters remain variables with scalar types and must retain their position in
the function signature.

## Adapt an IR Extension

| Previous access | Native contract |
| --- | --- |
| `PrimFunc.buffer_map` | Inspect each parameter's `ty`; tensor parameters carry `tirx.TensorType`. |
| A single arbitrary `func.body` statement | `Function.body` is an `ir.SeqStmt`; inspect its `seq` explicitly. |
| Buffer dtype, shape, scope, alignment | Read `tensor.ty.dtype`, `shape`, `scope`, and `data_alignment`. |
| Buffer strides and element offset | Inspect the type's layout and the tensor's declaration/view bindings. Strides and offsets are not independent `TensorType` fields. |
| Buffer data variable | `tirx.tensor_data_ptr(tensor)` projects a typed data pointer; view origin and pointer offset still matter. |
| `BufferLoad` / `BufferStore` | `ir.TensorLoad` / `ir.TensorStore`, referencing the actual tensor variable. |
| `prim_func_pass` | `tirx.transform.function_pass` receives and returns native `Function` objects. |
| Legacy object visitor | Use the native visitor/mutator interfaces or `tvm_ffi.structural_walk` / `structural_mutate`. |

When changing a tensor's shape, layout, or type, rebuild its definition and all
uses together. Preserve variable identity, lexical scope, and definitions used
inside types. Renaming a variable or replacing only a load's tensor field does
not perform that transformation. Validate the result with
`tirx.analysis.check_well_formed` and `tirx.analysis.undefined_vars`; use structural
equality and hashing from the same FFI provider when checking IR round trips.

TileLang-specific calls and regions carry explicit operands plus metadata
schemas. A native `ir.Call` has `args` and `attrs`, not the previous TileLang
`Call.annotations` field. Do not interpret every argument as an operation input
without checking the operation's operand and metadata contract.

## Compiler and Cache Identity

Use the matched TileLang, compiler provider, and FFI packages described in the
[installation guide](../get_started/Installation.md). Old binary caches cannot be
reused across the IR and ABI change. Keep the compiler source identity, target,
compile options, parameter signature, and output indices in specialization and
cache keys. Custom pass modules must register against the provider loaded by the
current process.
