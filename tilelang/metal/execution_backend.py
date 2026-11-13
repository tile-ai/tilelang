from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec


# tvm_ffi is registered first so that `execution_backend="auto"` resolves to it.
# The torch adapter (torch.mps.compile_shader) binds arguments positionally rather
# than in the emitted [[buffer(N)]] order, cannot launch a kernel whose grid depends
# on a T.dynamic() extent, and ignores out_idx; tvm_ffi is correct on all three and
# is no slower. torch remains available as an explicit opt-in.
EXECUTION_BACKENDS = [
    ExecutionBackendSpec("tvm_ffi", enable_host_codegen=True, enable_device_compile=True),
    ExecutionBackendSpec("torch"),
]
