import numpy as np
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from examples.dequantize_gemm.quantize.quantization import _tir_packed_to_unsigned_convert_with_zeros
from tvm import tirx


@tilelang.testing.requires_cuda
def test_packed_unsigned_convert_with_zeros_uses_signed_output_domain():
    """Zero-point subtraction must not underflow in uint32 packed storage."""

    n = 16
    zero = 8
    convert = _tir_packed_to_unsigned_convert_with_zeros("uint", 32)

    @T.prim_func
    def dequantize(packed: T.Tensor((n,), "uint32"), decoded: T.Tensor((n,), "float16")):
        with T.Kernel(1, threads=n):
            thread_id = T.get_thread_binding()
            decoded[thread_id] = convert(
                4,
                packed[thread_id],
                tirx.const(0, "int32"),
                tirx.const(zero, "uint32"),
                "float16",
            )

    kernel = tilelang.compile(dequantize, target="cuda", execution_backend="nvrtc")
    packed = torch.tensor(np.arange(n, dtype=np.uint32), device="cuda")
    decoded = torch.empty(n, dtype=torch.float16, device="cuda")
    kernel(packed, decoded)

    expected = torch.arange(n, dtype=torch.float16) - zero
    torch.testing.assert_close(decoded.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
