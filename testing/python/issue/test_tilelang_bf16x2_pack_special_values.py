import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from examples.dequantize_gemm.quantize.quantization import (
    _tir_f32x2_to_bf16x2_to_u32,
    _tir_u32_to_bf16x2_to_f32x2,
)


@tilelang.testing.requires_cuda
def test_f32x2_to_bf16x2_preserves_nan_and_inf():
    """RNE bf16 packing must not round inf/NaN bit patterns."""

    @T.prim_func
    def main(bits: T.Tensor((4,), "uint32"), out: T.Tensor((4,), "float32")):
        with T.Kernel(2, threads=1) as bx:
            f0 = T.reinterpret(bits[bx * 2], T.float32)
            f1 = T.reinterpret(bits[bx * 2 + 1], T.float32)
            packed = _tir_f32x2_to_bf16x2_to_u32(f0, f1)
            decoded = list(_tir_u32_to_bf16x2_to_f32x2(packed))
            out[bx * 2] = decoded[0]
            out[bx * 2 + 1] = decoded[1]

    kernel = tilelang.compile(main, target="cuda")
    bits = torch.tensor(
        [0x7FFF8C22, 0x7FFFFFFF, 0x7F800000, 0xFF800000],
        dtype=torch.uint32,
        device="cuda",
    )
    out = torch.empty(4, dtype=torch.float32, device="cuda")
    kernel(bits, out)

    assert torch.isnan(out[:2]).all(), out.cpu().tolist()
    assert torch.isinf(out[2:]).all(), out.cpu().tolist()
    assert out[2].item() > 0
    assert out[3].item() < 0


if __name__ == "__main__":
    tilelang.testing.main()
