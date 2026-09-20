from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec

# Plain Ascend prefers tvm_ffi -- it is listed first, so execution_backend="auto"
# resolves to it -- and also accepts cython.
ASCEND_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        enable_host_codegen=True,
        enable_device_compile=True,
        # Ascend's host codegen lowers the TVM-FFI callee-allocated-output
        # result slot, so a kernel with out_idx allocates its outputs inside
        # the generated host function. This is the single gate for that ABI
        # (see tilelang/jit/abi.py): MakePackedAPI and the adapter both read
        # the attribute it stamps, so the backend declares the capability here
        # instead of each side sniffing the target.
        supports_callee_allocated_outputs=True,
    ),
    ExecutionBackendSpec("cython"),
]

# PTO emits a source-only module with no runnable tvm_ffi runtime, so it must use
# the cython AOT pipeline (ptodsl -> ptoas -> bisheng).
PTO_EXECUTION_BACKENDS = [
    ExecutionBackendSpec("cython"),
]
