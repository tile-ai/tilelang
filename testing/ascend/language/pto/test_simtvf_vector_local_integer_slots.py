"""Vector local slots preserve byte integer values across mutation."""

import pytest
import torch
import torch_npu  # noqa: F401

import tilelang
import tilelang.testing
import tilelang.ascend.language as T


def _kernel(dtype):
    @T.prim_func
    def main(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1):
            src = T.alloc_shared((128,), dtype)
            dst = T.alloc_shared((128,), dtype)
            T.copy(A, src)
            with T.SimtVF(threads=1):
                local = T.alloc_local((3,), f"{dtype}x32")
                local[0] = src[T.Ramp(0, 1, 32)]
                local[1] = src[T.Ramp(32, 1, 32)]
                local[2] = src[T.Ramp(64, 1, 32)]
                dst[T.Ramp(0, 1, 32)] = T.min(local[0], local[1])
                local[1] = src[T.Ramp(96, 1, 32)]
                dst[T.Ramp(32, 1, 32)] = local[0]
                dst[T.Ramp(64, 1, 32)] = local[1]
                dst[T.Ramp(96, 1, 32)] = local[2]
            T.copy(dst, C)

    return main


@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_vector_local_byte_integer_slot_codegen(dtype):
    from tilelang.ascend.target import normalize_pto_target

    target = normalize_pto_target("pto")
    with target:
        source = tilelang.lower(_kernel(dtype), target=target, enable_device_compile=False).kernel_source
    logical = "pto.si8" if dtype == "int8" else "pto.ui8"
    assert "pto.alloc_buffer((96,), pto.i8)" in source
    assert "local[" not in source
    for offset in (0, 32, 64):
        assert f"pto.cast(pto.load(local, {offset}, contiguous=32), {logical})" in source
        assert f", pto.i8), local, {offset}, contiguous=32)" in source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("dtype", ["int8", "uint8"])
def test_vector_local_byte_integer_slot_mutation(dtype):
    kernel = tilelang.compile(_kernel(dtype), target="pto", out_idx=-1)
    torch_dtype = torch.int8 if dtype == "int8" else torch.uint8
    # High-bit bytes distinguish signed and unsigned minimum. Mutating the
    # middle slot must preserve both neighboring slots.
    a = torch.tensor([128, 255, 0, 127] * 8 + [1, 2, 254, 126] * 8 + [9, 10, 11, 12] * 8 + [40, 41, 42, 43] * 8).to(torch_dtype)
    expected = torch.cat((torch.minimum(a[:32], a[32:64]), a[:32], a[96:128], a[64:96]))
    result = kernel(a.to("npu"))
    torch.npu.synchronize()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
