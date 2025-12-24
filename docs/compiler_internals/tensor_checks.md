# Tensor Checks (Host-Side Auto-Validation)

This page explains the host-side checks that TileLang automatically inserts into the generated host stub for kernels. When you pass `torch.Tensor` or any DLPack-compatible object to a TileLang kernel, the host stub validates argument count, pointer kinds, dtype, shape, strides, device, and more — so you don’t need to handwrite Python checks. This keeps the ABI stable and significantly reduces Python overhead compared to doing equivalent checks in Python or via pybind.

## Why Host-Side Checks
- ABI stability: the entry is based on TVM FFI + DLPack, consistently accepting tensors and scalars.
- Lower overhead: shifting checks from Python into C reduces interpreter/property-access costs; the call overhead is lower than pybind-based approaches.
- Focused error reporting: assertions are raised close to the call site with precise “which field failed” messages.

## How To Inspect Host Source
You can inspect the auto-generated host source (with all checks and the final device-kernel call) for debugging:

```python
print(matmul_relu_kernel.get_host_source())
```

---

## What The Host Checks

### 1) Argument count and pointer kind
- `num_args` must match the number of formal parameters; otherwise the kernel returns `-1` with an error message.
- Each argument’s FFI type must be a pointer kind (for DLTensor/handle) or a valid scalar type; otherwise you’ll see errors like `Expect arg[i] to be pointer` or a scalar type error.

### 2) Tensor checks (per tensor, after nullability decision)
- Nullability
  - If the tensor is “statically reachable/used” by the function body, the handle must be non-NULL; otherwise: `xxx is expected to have non-NULL pointer`.
  - If an input tensor is not used by the function (statically unreachable), NULL is allowed; other field checks are executed only when `handle != NULL`.
- Rank (`ndim`)
  - Runtime `ndim` must equal the compile-time rank.
- Data type (`dtype`)
  - Match the triple `(code, bits, lanes)` with tolerance:
    - `float8_e4m3`: accept `e4m3`, `e4m3fn`, `e4m3fnuz`.
    - `float8_e5m2`: accept `e5m2`, `e5m2fnuz`.
    - `bool`: accept `int8/uint8` with `bits=8` (same lanes), `kDLBool(code=6, bits=1 or 8)`, and any `bitwidth=1` (lanes must match).
  - For packed-bit dtypes (e.g., `Int(1)`, `Int(4)`, `UInt(4)`), strict dtype checking is skipped.
- Shape
  - Each runtime dimension is bound to the compile-time shape (constants or symbols) and checked for consistency.
  - Linear equations among symbolic dims can be solved on the fly (when there’s only one unknown at a given check point), enabling cross-tensor constraints.
- Strides
  - If `buffer_type = AutoBroadcast`: allow `strides == NULL` and derive strides from `shape`. If explicit `strides` is present, bind to compile-time constraints and check for equality.
  - Otherwise: check per-dimension; if `strides == NULL`, derive from `shape` and compare (e.g., contiguous: `strides[-1] == 1`, `strides[-2] == shape[-1]`).
- `byte_offset`
  - Must be 0 (non-zero raises an error) to keep addressing simple and aligned.
- Device info
  - Assert `device_type == target backend` (CUDA/ROCM/Metal/OneAPI/WebGPU/CPU, etc.). Error messages include a DLPack code legend.
  - When multiple tensors participate, assert that `device_id` matches across them.
- Data pointer
  - Must be non-NULL when the tensor is required to be non-null by the nullability rule.

### 3) Scalar checks
- `T.int*` family: require integer; error: `Expect arg[i] to be int`.
- `T.bool`: require boolean; error: `Expect arg[i] to be boolean`.

---

## Shapes and Symbolic Equations: Linear Solving
When shapes are symbolic, the host binds and (when possible) solves linear relations at runtime (only one unknown per check point). Example:

```python
@T.prim_func
def main(
    A: T.Tensor((m,), dtype),
    B: T.Tensor((m + n,), dtype),
    C: T.Tensor((n * k,), dtype),
):
    ...
```

This enables enforcing cross-tensor relationships like `len(B) == m + n` and `len(C) == n * k` at runtime.

---

## Nullability Rules and Examples
Which tensors may be NULL?

- Rule: If an input tensor is not used by the function under static analysis (i.e., the access is statically unreachable), it is considered Nullable; otherwise it must be non-NULL.
- Examples:

1) Must be non-NULL (used)
```python
@T.prim_func
def main(A: T.Tensor((M, K), dtype)):
    A[0] = 1
```
Passing `None` raises: `main.A_handle is expected to have non-NULL pointer`.

2) Still must be non-NULL (constant-true branch)
```python
some_cond: bool = True
@T.prim_func
def main(A: T.Tensor((M, K), dtype)):
    if some_cond:
        A[0] = 1
```

3) Nullable (constant-false branch, statically unreachable)
```python
some_cond: bool = False
@T.prim_func
def main(A: T.Tensor((M, K), dtype)):
    if some_cond:
        A[0] = 1
```

4) Must be non-NULL (runtime condition)
```python
@T.prim_func
def main(A: T.Tensor((M, K), dtype), some_cond: T.bool):
    if some_cond:
        A[0] = 1
```
Since `some_cond` is only known at runtime, static analysis cannot prove `A` is unused; `A` is thus non-nullable.

---

## Device Type Codes (DLPack)
Supported and referenced device codes in error messages: `1=CPU, 2=CUDA, 7=Vulkan, 8=Metal, 10=ROCM, 14=OneAPI, 15=WebGPU`.
Kernels assert that `device_type` matches the target backend, and require `device_id` consistency across tensors.

---

## Common Error Examples (What you’ll see)
- Argument count mismatch (num_args)
  - Trigger: missing/extra argument
  - Error: `<kernel>: num_args should be N; expected: <num_args>, got: N`

- Pointer-typed argument expected
  - Trigger: scalar passed where a tensor is expected
  - Error: `<kernel>: Expect arg[i] to be pointer`

- Rank (ndim) mismatch
  - Trigger: runtime rank differs from compile-time rank
  - Error: `<kernel>.<name>.ndim is expected to equal R, but got mismatched ndim`

- Dtype mismatch
  - Trigger: dtype not equal to the compiled dtype and not within the tolerance set
  - Error: `<kernel>.<name>.dtype is expected to be <dtype>, but got incompatible dtype`

- Shape constraint violation
  - Trigger: a dimension doesn’t match a constant/symbol binding
  - Error: `Argument <kernel>.<name>.shape[i] has an unsatisfied constraint: ... == <expected>`

- Strides check failed (e.g., non-contiguous layout)
  - Trigger: transposed/sliced tensors that violate expected strides
  - Error: `Argument <kernel>.<name>.strides[j] has an unsatisfied constraint: ... == <expected>`

- Device type mismatch
  - Trigger: calling a CUDA kernel with CPU tensors, etc.
  - Error: `<kernel>.<name>.device_type mismatch [expected: <code> (<name>)] ...`

- Device id mismatch
  - Trigger: mixing tensors from different GPUs
  - Error: `Argument <kernel>.<name>.device_id has an unsatisfied constraint: ... == ...`

- NULL data pointer
  - Trigger: tensor required to be non-null has a NULL data pointer
  - Error: `<kernel>.<name> is expected to have non-NULL data pointer, but got NULL`

- Scalar type mismatch
  - Trigger: passing float to `T.int32`, or non-boolean to `T.bool`
  - Error: `<kernel>: Expect arg[i] to be int/boolean`

---

## Troubleshooting Tips
- Print the host source: `print(fn.get_host_source())` to see the exact assertion and expected vs. actual fields.
- Fix strides: call `.contiguous()` for non-contiguous tensors, or avoid generating transposed/sliced layouts that break assumptions.
- Align devices: ensure all participating tensors share the same `device_type` and `device_id`.
- Align dtype: use `.to(<dtype>)` or construct tensors with the correct dtype; pay attention to `float8` and `bool` tolerance.
- Dynamic shapes: ensure cross-tensor linear relations can be uniquely determined at the check point (only one unknown at a time).

---

## FAQ
- Can I disable the checks?
  - Not recommended and usually not supported. Checks are done on the host to preserve ABI stability and fail early close to the device call.
- Is the overhead noticeable?
  - The checks are lightweight (branches and field reads). Compared to Python-side checks, it’s faster; the dominating cost remains the Python→C boundary. Overall it’s cheaper than equivalent checks in Python.

---

## Reference Example (Matmul + ReLU)

```python
@T.prim_func
def matmul_relu_kernel(
    A: T.Tensor((M, K), dtype),
    B: T.Tensor((K, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    # Initialize Kernel Context
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])

# For debugging, print the host source
print(matmul_relu_kernel.get_host_source())
```

The host will insert all checks described above for this example.

---

## Quick Error Reference (Short List)
- Argument count
  - Trigger: missing/extra args; Error: `num_args should be N; expected: <num_args>, got: N`.
- Pointer kind
  - Trigger: scalar passed to tensor arg; Error: `Expect arg[i] to be pointer`.
- Rank (ndim)
  - Trigger: runtime rank != compile-time; Error: `ndim ... expected to equal R`.
- Dtype
  - Trigger: mismatch and not tolerated; Error: `dtype ... expected to be <dtype>`.
- Shape
  - Trigger: constant/symbol binding violated; Error: `shape[i] ... == <expected>`.
- Strides
  - Trigger: layout mismatch; Error: `strides[j] ... == <expected>`.
- Device type
  - Trigger: wrong backend device; Error: `device_type mismatch [expected: ...]`.
- Device id
  - Trigger: tensors on different GPUs; Error: `device_id ... == ...`.
- Data pointer
  - Trigger: required non-NULL but NULL; Error: `non-NULL data pointer`.
- Scalar types
  - Trigger: wrong scalar type; Error: `Expect arg[i] to be int/boolean`.

---

## Host Error Troubleshooting (Minimal Repros)

Below are minimal repro snippets for common host-side errors, assuming a CUDA-targeted kernel like `matmul_relu_kernel` with:

```python
# Convention:
# A: float16 [M, K]
# B: float16 [K, N]
# C: float16 [M, N]
# Target: CUDA (device_type=2)
fn = matmul_relu_kernel  # your compiled function
M = N = K = 1024
```

Adjust dtype/device if your kernel differs.

### 0. Tip: print the host source
```python
print(fn.get_host_source())
```

### 1. num_args mismatch
```python
import torch

A = torch.empty((M, K), device='cuda', dtype=torch.float16)
B = torch.empty((K, N), device='cuda', dtype=torch.float16)
# Missing C
fn(A, B)
```
Expected: `<kernel>: num_args should be 3; expected: <num_args>, got: 3`.

Fix: pass all arguments per the signature.

### 2. Expect pointer (tensor) but got scalar
```python
import torch

B = torch.empty((K, N), device='cuda', dtype=torch.float16)
C = torch.empty((M, N), device='cuda', dtype=torch.float16)
fn(1, B, C)
```
Expected: `<kernel>: Expect arg[0] to be pointer`.

Fix: pass a DLPack-compatible tensor (e.g., torch.Tensor).

### 3. ndim mismatch
```python
import torch

A = torch.empty((M, K, 1), device='cuda', dtype=torch.float16)  # rank=3
B = torch.empty((K, N), device='cuda', dtype=torch.float16)
C = torch.empty((M, N), device='cuda', dtype=torch.float16)
fn(A, B, C)
```
Expected: `<kernel>.A_handle.ndim is expected to equal 2, but got mismatched ndim`.

Fix: ensure runtime rank equals compiled rank.

### 4. dtype mismatch
```python
import torch

A = torch.empty((M, K), device='cuda', dtype=torch.float32)  # should be float16
B = torch.empty((K, N), device='cuda', dtype=torch.float16)
C = torch.empty((M, N), device='cuda', dtype=torch.float16)
fn(A, B, C)
```
Expected: `<kernel>.A_handle.dtype is expected to be float16, but got incompatible dtype`.

Fix: `A = A.to(torch.float16)` or create with the correct dtype.

### 5. Shape constant/symbol mismatch
```python
import torch

A = torch.empty((M, K + 1), device='cuda', dtype=torch.float16)  # K mismatched
B = torch.empty((K, N), device='cuda', dtype=torch.float16)
C = torch.empty((M, N), device='cuda', dtype=torch.float16)
fn(A, B, C)
```
Expected: `Argument <kernel>.A_handle.shape[i] has an unsatisfied constraint: ... == <expected>`.

Fix: satisfy linear constraints and constants across tensors.

### 6. Strides check failure (non-contiguous)
```python
import torch

A = torch.empty((M, K), device='cuda', dtype=torch.float16)
A_nc = A.t()  # transpose -> non-contiguous
B = torch.empty((K, N), device='cuda', dtype=torch.float16)
C = torch.empty((M, N), device='cuda', dtype=torch.float16)
fn(A_nc, B, C)
```
Expected: `Argument <kernel>.A_handle.strides[1] has an unsatisfied constraint: ... == 1`.

Fix: pass `A_nc.contiguous()` or align the layout expectation in the kernel.

### 7. device_type mismatch
```python
import torch

A = torch.empty((M, K), device='cpu', dtype=torch.float16)
B = torch.empty((K, N), device='cpu', dtype=torch.float16)
C = torch.empty((M, N), device='cpu', dtype=torch.float16)
fn(A, B, C)  # CUDA-targeted kernel
```
Expected: `<kernel>.A_handle.device_type mismatch [expected: 2 (cuda)] ...`.

Fix: move tensors to the CUDA device.

### 8. device_id mismatch (multi-GPU)
```python
import torch

A = torch.empty((M, K), device='cuda:0', dtype=torch.float16)
B = torch.empty((K, N), device='cuda:1', dtype=torch.float16)
C = torch.empty((M, N), device='cuda:0', dtype=torch.float16)
fn(A, B, C)
```
Expected: `Argument <kernel>.B_handle.device_id has an unsatisfied constraint: ... == ...`.

Fix: place all tensors on the same GPU (e.g., `cuda:0`).

### 9. NULL data pointer (advanced)
This usually comes from hand-constructed DLTensor/NDArray, or external frameworks passing unallocated/freed storage. Regular `torch.Tensor` allocations rarely hit this.

Expected: `<kernel>.<name> is expected to have non-NULL data pointer, but got NULL`.

Fix: ensure valid underlying storage; in PyTorch scenarios, avoid constructing tensors from invalid external handles.

### 10. Scalar type mismatch (int / bool)
```python
import tilelang.language as T

@T.prim_func
def scalar_check(x: T.int32, flag: T.bool()):
    T.evaluate(0)

scalar_check(1.0, True)  # x is float -> Expect arg[0] to be int
scalar_check(1, 2.5)     # flag is float -> Expect arg[1] to be boolean
```

Fix: pass correct scalar types, e.g., `scalar_check(1, True)`.

---

## Closing Notes
- Cross-check “shape / strides / device / dtype” against the kernel signature to localize issues efficiently.
- For complex symbolic relations, print the host source to confirm binding/solving order, then adjust runtime shapes/layouts accordingly.
