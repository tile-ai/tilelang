"""Preserve integer cast semantics inside SIMT vector expressions."""

import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.testing


def _inline_unsigned_min_max():
    @T.prim_func
    def main(a: T.Tensor((4,), T.int32), out: T.Tensor((4,), T.int32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((4,), T.int32)
            out_ub = T.alloc_shared((4,), T.int32)
            T.copy(a, a_ub)
            with T.SimtVF(threads=1):
                out_ub[T.Ramp(0, 1, 2)] = T.Cast(
                    "int32x2",
                    T.min(T.Cast("uint32x2", a_ub[T.Ramp(0, 1, 2)]), T.Cast("uint32x2", a_ub[T.Ramp(2, 1, 2)])),
                )
                out_ub[T.Ramp(2, 1, 2)] = T.Cast(
                    "int32x2",
                    T.max(T.Cast("uint32x2", a_ub[T.Ramp(0, 1, 2)]), T.Cast("uint32x2", a_ub[T.Ramp(2, 1, 2)])),
                )
            T.copy(out_ub, out)

    return main


def _integer_vector_roundtrip(intermediate_dtype, lanes=2):
    num_elements = 2 * lanes
    out_dtype = f"int32x{lanes}"

    @T.prim_func
    def main(a: T.Tensor((num_elements,), T.int32), out: T.Tensor((num_elements,), T.int32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((num_elements,), T.int32)
            out_ub = T.alloc_shared((num_elements,), T.int32)
            T.copy(a, a_ub)
            with T.SimtVF(threads=2):
                tx = T.get_thread_binding()
                indices = T.Ramp(tx * lanes, 1, lanes)
                out_ub[indices] = T.Cast(out_dtype, T.Cast(intermediate_dtype, a_ub[indices]))
            T.copy(out_ub, out)

    return main


def test_pto_inline_integer_vector_cast_retains_unsigned_operands():
    source = tilelang.lower(_inline_unsigned_min_max(), target="pto", enable_device_compile=False).kernel_source
    # There is no unsigned storage whose load could restore a discarded cast.
    assert "pto.min(pto.cast(" in source
    assert "pto.max(pto.cast(" in source
    assert source.count(", pto.ui32)") >= 4


@pytest.mark.parametrize("intermediate_dtype,pto_dtype", [("int16x2", "i16"), ("uint16x2", "ui16")])
def test_pto_integer_vector_width_cast_is_explicit(intermediate_dtype, pto_dtype):
    source = tilelang.lower(_integer_vector_roundtrip(intermediate_dtype), target="pto", enable_device_compile=False).kernel_source
    assert f", pto.{pto_dtype})" in source
    assert ", pto.i32)" in source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", "pto"])
def test_inline_unsigned_vector_min_max(target):
    import torch

    values = torch.tensor([-2147483648, -1, 1, 1], dtype=torch.int32, device="npu")
    kernel = tilelang.compile(_inline_unsigned_min_max(), target=target, out_idx=-1)
    actual = kernel(values)
    expected = torch.tensor([1, 1, -2147483648, -1], dtype=torch.int32)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "intermediate_dtype,values,expected",
    [
        ("int16x2", [-65537, 65535, 32768, -32769], [-1, -1, -32768, 32767]),
        ("uint16x2", [-65537, 65535, 32768, -32769], [65535, 65535, 32768, 32767]),
        ("boolx2", [0, 2, -2, -1], [0, 1, 1, 1]),
    ],
)
def test_pto_integer_vector_cast_roundtrip(intermediate_dtype, values, expected):
    import torch

    x = torch.tensor(values, dtype=torch.int32, device="npu")
    kernel = tilelang.compile(_integer_vector_roundtrip(intermediate_dtype), target="pto", out_idx=-1)
    actual = kernel(x)
    torch.testing.assert_close(actual.cpu(), torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0)


@tilelang.testing.requires_ascend
def test_pto_wide_bool_vector_cast_normalizes_each_lane():
    import torch

    x = torch.tensor([0, 2, -2, -1, 1, 0, -2147483648, 2147483647], dtype=torch.int32, device="npu")
    kernel = tilelang.compile(_integer_vector_roundtrip("boolx4", lanes=4), target="pto", out_idx=-1)
    actual = kernel(x)
    expected = (x.cpu() != 0).to(torch.int32)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def _unsigned_reduce(op, extent):
    reduce = getattr(T, f"reduce_{op}")

    @T.prim_func
    def main(a: T.Tensor((2, extent), "uint32"), out: T.Tensor((2,), "uint32")):
        with T.Kernel(1), T.SimtVF(threads=2 * extent):
            values = T.alloc_fragment((2, extent), "uint32")
            result = T.alloc_fragment((2,), "uint32")
            T.copy(a, values)
            reduce(values, result, dim=1)
            T.copy(result, out)

    return main


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("extent", [8, 64])
@pytest.mark.parametrize("op", ["min", "max"])
def test_pto_unsigned_reduce_preserves_select_type(extent, op):
    import torch

    # Cover masked warp reduction and independent cross-warp groups. A lost
    # unsigned type reverses the order of the large values and the small ones.
    values = [[0xFFFFFFFF] + [1] * (extent - 1), [0x80000000] + [2] * (extent - 1)]
    reference = {"min": min, "max": max}[op]
    expected = torch.tensor([reference(row) for row in values], dtype=torch.uint32)
    x = torch.tensor(values, dtype=torch.uint32, device="npu")
    kernel = tilelang.compile(_unsigned_reduce(op, extent), target="pto", out_idx=-1)
    actual = kernel(x)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
