# Tensor Checks (Host-Side Auto-Validation)

TileLang's `tvm_ffi` execution path uses a native packed host entry to validate
runtime arguments against the compiled tensor and scalar contracts. A Python
adapter first converts supported framework objects through TVM FFI/DLPack;
other execution backends own their corresponding source-wrapper/runtime checks.
Do not assume every adapter produces identical host source or error wording.

## Where the Contract Lives

The source `Function` contains ordinary `Var` parameters. Tensor parameters
carry `TensorType` with shape, element type, scope and layout. There is no
`PrimFunc.buffer_map`. `src/transform/make_packed_api.cc` attaches TileLang's
packed-input policies and delegates to the native
`3rdparty/tvm-tirx/src/tirx/transform/tvm_ffi_binder.cc`.

That binder extracts all argument handles and safe descriptor fields before
resolving implicit dimensions. A shape variable may therefore be supplied by a
later parameter. Constraints are checked after the required bindings exist,
including strided layouts and their element offsets. Callee-allocated outputs
use TileLang's owning output frame around native allocation/compute workers.

## Inspecting Host Source

`@T.prim_func` constructs an IR `Function`. Compile it before accessing runtime
methods:

```python
import tilelang

kernel = tilelang.compile(program, target="cuda", execution_backend="tvm_ffi")
print(kernel.get_host_source())
```

The selected host backend may return C source or LLVM IR. Inspect the actual
argument names and assertions in that result; names from an older printed
stub, such as `A_handle`, are not a stable native IR field or diagnostic API.

## Checks at the Packed Entry

| Field | Contract |
| --- | --- |
| Argument count and kind | Match the selected packed worker's parameter list and its FFI value kinds. Output-allocation wrappers may provide output arguments internally. |
| Tensor descriptor | A used tensor requires a descriptor. A statically unused tensor may be `None` under TileLang's explicit nullable-input policy; descriptor loads are guarded. |
| Rank | Match the number of dimensions in the compiled `TensorType`. |
| Element type | Match code, bits and lanes after adapter normalization, with the explicit FP8 and packed-storage policies described below. |
| Shape | Bind symbolic dimensions and enforce constants/equalities across present arguments. |
| Strides | Match the tensor's shape-aligned strided layout. A null DLPack strides pointer means compact strides, derived from shape. |
| Byte offset | Match the layout's element offset after conversion to bytes. A default zero-origin layout requires zero; an explicit nonzero/symbolic offset is checked against that contract. |
| Device | Match the selected target's device type and the common device ID for participating tensors. |
| Data pointer | A present nonempty tensor needs non-null storage. A zero-size tensor may have a null data pointer. Direct-address devices also receive the required alignment checks. |

Native descriptor checks do not accept an arbitrary runtime layout for a
compact tensor. Use `T.StridedTensor` or an explicit supported tensor layout
when non-contiguous inputs are intentional. Calling `.contiguous()` is useful
only when it matches the intended input contract.

### FP8 and packed storage

TileLang requests the native binder's FP8 alias policy: logical
`float8_e4m3` also accepts the corresponding `fn`/`fnuz` storage codes, and
`float8_e5m2` accepts its `fnuz` alias, with matching bits and lanes. These
aliases do not imply support for every device instruction or numerical format.

For supported packed sub-byte inputs, byte storage is an explicit bit
representation. The binder expands the innermost physical dimension into a
logical dimension using the packing factor; outer strides and byte offsets
are converted into logical-element units.
Unit stride in the packed dimension, shape consistency, divisibility and range
checks still apply. The packed path does not simply skip all dtype/shape checks.
Framework bool aliases are normalized by the adapter; the native descriptor is
checked against the resulting logical tensor contract.

### Scalars

The native FFI binder accepts an integer or boolean FFI value for integral
parameters and for boolean parameters. Floating parameters accept floating,
integer or boolean FFI values and apply the declared primitive conversion.
Passing a floating FFI value to an integer parameter is a type error. The Python
JIT can additionally use scalar arguments for compile-time specialization;
those values need not be runtime packed slots.

## Dynamic Shapes and Nullability

Direct symbolic carriers can appear later in the parameter list. Supported
single-variable affine forms such as `2 * n + 1` can bind `n` after checking
divisibility. Compound expressions whose variables are already bound become
equality constraints. This is not a general nonlinear equation solver.

```python
@T.prim_func
def shape_contract(
    A: T.Tensor((n + 1,), "float32"),
    B: T.Tensor((n,), "float32"),
): ...
```

The ABI can read `n` from `B` before checking `A`, even though `B` is later in
the signature. This snippet illustrates a signature; a complete kernel supplies
the computation and its `n` declaration.

Nullable means that the compiled body does not require that tensor identity;
types, layouts and explicit annotation operands also participate in use
analysis. A runtime-false branch does not make its tensor statically unused.
Null descriptors cannot be dereferenced to recover shapes. When multiple
nullable carriers provide a symbol needed by the body, at least one must be
present, or the call reports the missing symbolic-shape source.

## Errors and Troubleshooting

Messages include the argument index, tensor name, failed field and function
signature. Representative native diagnostic fragments are:

- `Expected ... arguments` for arity mismatch;
- `Mismatched A.ndim on argument #...` for rank;
- `Mismatched A.dtype on argument #...` for element type;
- `Mismatched A.device_type on argument #...` for device placement;
- an access path such as `shape[i]`, `strides[i]` or `byte_offset` for an
  unsatisfied descriptor constraint;
- `expected non-NULL data pointer` for a present nonempty tensor without storage.

TileLang requests `RuntimeError` for tensor-contract failures from its native
packed workers; FFI kind/arity checks and Python adapters may raise other
appropriate exception types. Check the actual exception and host output instead
of depending on a complete old error string.

To isolate a failure, compare runtime shapes, strides, dtype, device and offset
with the compiled signature. Confirm the loaded provider/FFI identity after a
native rebuild, and use a supported DLPack owner so storage remains alive until
execution completes. Checks cannot make a dangling external pointer safe.

## Regression Coverage

Keep distinct tests for late dimension carriers, affine divisibility, conflicting
cross-tensor shapes, non-contiguous strides, nonzero byte offsets, packed-bit
storage, nullable descriptors, empty storage, output allocation and scalar kind
errors. Validate each supported execution adapter independently; source
generation, module loading and numerical execution are separate observations.
