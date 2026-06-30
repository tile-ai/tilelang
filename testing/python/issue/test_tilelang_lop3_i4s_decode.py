"""Regression coverage for signed int4 LOP3 decode."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.quantize import _tir_packed_int_to_int_convert, get_lop3_intrin_group
from tilelang.quantize.utils import interleave_weight


N = 16
K = 16
NUM_BITS = 4
STORAGE = T.int8
IN_DTYPE = T.int8
ELEMS_PER_BYTE = 8 // NUM_BITS
K_PACKED = K // ELEMS_PER_BYTE


def _build_scalar_signed_decode():
    @T.prim_func
    def main(B: T.Tensor((N, K_PACKED), STORAGE), D: T.Tensor((N, K), IN_DTYPE)):
        with T.Kernel(N, threads=32) as bx:
            B_local = T.alloc_local([K_PACKED], STORAGE)
            for v in T.serial(K_PACKED):
                B_local[v] = B[bx, v]
            for ki in T.serial(K):
                D[bx, ki] = _tir_packed_int_to_int_convert("int", 8)(NUM_BITS, B_local[ki // ELEMS_PER_BYTE], ki % ELEMS_PER_BYTE, IN_DTYPE)

    return main


def _build_lop3_decode(source_format):
    lop3 = get_lop3_intrin_group(
        out_dtype=IN_DTYPE,
        source_format=source_format,
        source_bit=NUM_BITS,
        storage_dtype=STORAGE,
        with_scaling=False,
        with_zeros=False,
    )
    source = lop3["c_source"]
    func_name = lop3["func_name"]
    group = 16
    group_bytes = group // ELEMS_PER_BYTE

    @T.prim_func
    def main(B: T.Tensor((N, K_PACKED), STORAGE), D: T.Tensor((N, K), IN_DTYPE)):
        with T.Kernel(N, threads=32) as bx:
            B_local = T.alloc_local([group_bytes], STORAGE)
            D_local = T.alloc_local([group], IN_DTYPE)
            T.import_source(source)
            for g in T.serial(K // group):
                for v in T.vectorized(group_bytes):
                    B_local[v] = B[bx, g * group_bytes + v]
                T.evaluate(T.call_extern("handle", func_name, T.access_ptr(B_local, "r"), T.access_ptr(D_local, "w")))
                for v in T.serial(group):
                    D[bx, g * group + v] = D_local[v]

    return main


def _pack_repeated_nibbles() -> torch.Tensor:
    nibbles = torch.arange(16, dtype=torch.uint8).repeat(N, 1)
    packed = nibbles[:, ::2] | (nibbles[:, 1::2] << 4)
    return packed.contiguous().view(torch.int8).cuda()


def _unpack_int4(qweight: torch.Tensor, signed: bool) -> torch.Tensor:
    qweight = qweight.to(torch.uint8)
    low = (qweight & 0x0F).to(torch.int8)
    high = (qweight >> 4).to(torch.int8)
    if signed:
        low = (low << 4) >> 4
        high = (high << 4) >> 4
    return torch.stack((low, high), dim=-1).reshape(qweight.shape[0], qweight.shape[1] * 2)


@tilelang.testing.requires_cuda
def test_lop3_i4s_to_i8s_matches_scalar_signed_decode():
    qweight = _pack_repeated_nibbles()
    expected_signed = _unpack_int4(qweight, signed=True)
    expected_unsigned = _unpack_int4(qweight, signed=False)

    scalar_kernel = tilelang.compile(_build_scalar_signed_decode(), target="cuda", out_idx=[1])
    scalar = scalar_kernel(qweight)
    torch.testing.assert_close(scalar, expected_signed, rtol=0, atol=0)

    lop3_signed_kernel = tilelang.compile(_build_lop3_decode(T.int), target="cuda", out_idx=[1])
    qweight_interleaved = interleave_weight(qweight.clone(), NUM_BITS, "int8")
    lop3_signed = lop3_signed_kernel(qweight_interleaved)
    torch.testing.assert_close(lop3_signed, scalar, rtol=0, atol=0)
    torch.testing.assert_close(lop3_signed, expected_signed, rtol=0, atol=0)

    lop3_unsigned_kernel = tilelang.compile(_build_lop3_decode(T.uint), target="cuda", out_idx=[1])
    lop3_unsigned = lop3_unsigned_kernel(qweight_interleaved)
    torch.testing.assert_close(lop3_unsigned, expected_unsigned, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
