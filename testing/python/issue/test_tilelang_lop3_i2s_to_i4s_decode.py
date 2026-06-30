"""Regression coverage for signed int2 to int4 LOP3 decode."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.quantize import get_lop3_intrin_group


N = 16
I2_STORAGE_BYTES = N // 4
I4_STORAGE_BYTES = N // 2


def _build_lop3_i2_to_i4_decode(source_format):
    lop3 = get_lop3_intrin_group(
        out_dtype=T.int4,
        source_format=source_format,
        source_bit=2,
        storage_dtype=T.int8,
        with_scaling=False,
        with_zeros=False,
    )
    source = lop3["c_source"]
    func_name = lop3["func_name"]

    @T.prim_func
    def main(I: T.Tensor((I2_STORAGE_BYTES,), T.int8), O: T.Tensor((I4_STORAGE_BYTES,), T.int8)):
        with T.Kernel(1, threads=1):
            T.import_source(source)
            i2 = T.alloc_local((I2_STORAGE_BYTES,), T.int8)
            i4 = T.alloc_local((I4_STORAGE_BYTES,), T.int8)
            for k in T.serial(I2_STORAGE_BYTES):
                i2[k] = I[k]
            T.evaluate(T.call_extern("handle", func_name, T.address_of(i2[0]), T.address_of(i4[0]), N))
            for k in T.serial(I4_STORAGE_BYTES):
                O[k] = i4[k]

    return main


def _pack_i2_values(values):
    values = torch.tensor(values, dtype=torch.uint8).reshape(-1, 4)
    packed = values[:, 0] | (values[:, 1] << 2) | (values[:, 2] << 4) | (values[:, 3] << 6)
    return packed.contiguous().view(torch.int8).cuda()


def _unpack_i4_values(packed, signed):
    packed = packed.to(torch.uint8)
    low = (packed & 0x0F).to(torch.int8)
    high = (packed >> 4).to(torch.int8)
    if signed:
        low = (low << 4) >> 4
        high = (high << 4) >> 4
    return torch.stack((low, high), dim=-1).reshape(-1)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(7, 5)
def test_lop3_i2s_to_i4s_sign_extends_int2():
    qweight = _pack_i2_values([0, 1, 2, 3] * 4)

    signed_kernel = tilelang.compile(_build_lop3_i2_to_i4_decode(T.int), target="cuda")
    unsigned_kernel = tilelang.compile(_build_lop3_i2_to_i4_decode(T.uint), target="cuda")

    signed = torch.empty(I4_STORAGE_BYTES, dtype=torch.int8, device="cuda")
    unsigned = torch.empty_like(signed)
    signed_kernel(qweight, signed)
    unsigned_kernel(qweight, unsigned)
    torch.cuda.synchronize()

    unsigned_values = _unpack_i4_values(unsigned, signed=False)
    signed_values = _unpack_i4_values(signed, signed=True)
    expected_signed = (unsigned_values << 6) >> 6

    torch.testing.assert_close(signed_values, expected_signed, rtol=0, atol=0)
    assert not torch.equal(signed, unsigned)
    assert torch.any(unsigned_values == 2)
    assert torch.any(unsigned_values == 3)


if __name__ == "__main__":
    tilelang.testing.main()
