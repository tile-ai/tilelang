import numpy as np
import pytest

import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.quantize import _tir_u8_to_f4_to_bf16
from tvm import tirx


@pytest.mark.parametrize(
    ("scale", "expected_bits"),
    [
        (
            0,
            [
                0x0000,
                0x3F00,
                0x3F80,
                0x3FC0,
                0x4000,
                0x4040,
                0x4080,
                0x40C0,
                0x8000,
                0xBF00,
                0xBF80,
                0xBFC0,
                0xC000,
                0xC040,
                0xC080,
                0xC0C0,
            ],
        ),
        (
            1,
            [
                0x0000,
                0x3F80,
                0x4000,
                0x4040,
                0x4080,
                0x40C0,
                0x4100,
                0x4140,
                0x8000,
                0xBF80,
                0xC000,
                0xC040,
                0xC080,
                0xC0C0,
                0xC100,
                0xC140,
            ],
        ),
    ],
)
@tilelang.testing.requires_llvm
def test_u8_to_fp4_e2m1_to_bf16_all_encodings(scale, expected_bits):
    @T.prim_func
    def decode(packed: T.Buffer((8,), T.uint8), decoded_bits: T.Buffer((16,), T.uint16)):
        for i in T.serial(16):
            value = _tir_u8_to_f4_to_bf16(
                4,
                packed[i // 2],
                i % 2,
                tirx.const(scale, T.uint16),
                T.bfloat16,
            )
            decoded_bits[i] = tirx.reinterpret(T.uint16, value)

    packed = tvm.nd.array(np.array([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE], dtype="uint8"))
    decoded_bits = tvm.nd.empty((16,), T.uint16, tvm.cpu())
    expected_bits = np.array(expected_bits, dtype="uint16")

    tvm.compile(decode, target="llvm")(packed, decoded_bits)

    np.testing.assert_array_equal(decoded_bits.numpy(), expected_bits)


if __name__ == "__main__":
    tilelang.testing.main()
