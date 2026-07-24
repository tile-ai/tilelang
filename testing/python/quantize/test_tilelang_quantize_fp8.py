import numpy as np

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from examples.dequantize_gemm.quantize.quantization import (
    _tir_u8_to_f8_e4m3_to_f16,
    _tir_u8_to_f8_e4m3_to_f16_naive,
)


def _e4m3_to_float16(code: int) -> np.float16:
    sign = -1.0 if code & 0x80 else 1.0
    exponent = (code >> 3) & 0xF
    mantissa = code & 0x7

    if exponent == 0:
        return np.float16(sign * np.ldexp(mantissa / 8.0, -6))
    if exponent == 0xF and mantissa == 0x7:
        return np.float16(np.nan)
    return np.float16(sign * np.ldexp(1.0 + mantissa / 8.0, exponent - 7))


@tilelang.testing.requires_llvm
def test_u8_to_f8_e4m3_to_f16_all_codes():
    @T.prim_func
    def decode(
        values: T.Tensor((256,), "uint8"),
        naive: T.Tensor((256,), "uint16"),
        optimized: T.Tensor((256,), "uint16"),
    ):
        for i in T.serial(256):
            naive[i] = T.reinterpret(_tir_u8_to_f8_e4m3_to_f16_naive(8, values[i], T.float16), T.uint16)
            optimized[i] = T.reinterpret(_tir_u8_to_f8_e4m3_to_f16(8, values[i], T.float16), T.uint16)

    built = tvm.compile(decode, target="llvm")
    values = tvm.runtime.tensor(np.arange(256, dtype=np.uint8))
    naive = tvm.runtime.empty((256,), "uint16", tvm.cpu())
    optimized = tvm.runtime.empty((256,), "uint16", tvm.cpu())
    built(values, naive, optimized)

    naive_bits = naive.numpy()
    optimized_bits = optimized.numpy()
    expected = np.array([_e4m3_to_float16(code) for code in range(256)], dtype=np.float16)
    finite = ~np.isnan(expected)

    np.testing.assert_array_equal(naive_bits[finite], expected[finite].view(np.uint16))
    np.testing.assert_array_equal(optimized_bits[finite], expected[finite].view(np.uint16))
    np.testing.assert_array_equal(np.isnan(naive_bits.view(np.float16)), np.isnan(expected))
    np.testing.assert_array_equal(np.isnan(optimized_bits.view(np.float16)), np.isnan(expected))
    np.testing.assert_array_equal(naive_bits, optimized_bits)


if __name__ == "__main__":
    tilelang.testing.main()
