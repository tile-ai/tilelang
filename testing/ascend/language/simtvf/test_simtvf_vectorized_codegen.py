"""Regression tests for small-vector expression codegen in Ascend SIMT blocks."""

import re

import pytest

import tilelang
import tilelang.testing
import tilelang.ascend.language as T


def _source(func, *, target, pass_configs=None):
    with tilelang.tvm.target.Target(target), tilelang.transform.PassContext(config=pass_configs or {}):
        return tilelang.lower(func, target=target).kernel_source


def _vector_min_max_mod_kernel():
    num_elements = 64

    @T.prim_func
    def main(
        a_int: T.Tensor((num_elements,), T.int32),
        b_int: T.Tensor((num_elements,), T.int32),
        a_float: T.Tensor((num_elements,), T.float32),
        b_float: T.Tensor((num_elements,), T.float32),
        out_int: T.Tensor((num_elements * 3,), T.int32),
        out_float: T.Tensor((num_elements * 2,), T.float32),
    ):
        with T.Kernel(1):
            a_int_ub = T.alloc_shared((num_elements,), T.int32)
            b_int_ub = T.alloc_shared((num_elements,), T.int32)
            a_float_ub = T.alloc_shared((num_elements,), T.float32)
            b_float_ub = T.alloc_shared((num_elements,), T.float32)
            out_int_ub = T.alloc_shared((num_elements * 3,), T.int32)
            out_float_ub = T.alloc_shared((num_elements * 2,), T.float32)
            T.copy(a_int, a_int_ub)
            T.copy(b_int, b_int_ub)
            T.copy(a_float, a_float_ub)
            T.copy(b_float, b_float_ub)
            with T.SimtVF(threads=32):
                lane = T.get_thread_binding()
                for i in T.vectorized(2):
                    index = 2 * lane + i
                    out_int_ub[index] = T.min(a_int_ub[index], b_int_ub[index])
                    out_int_ub[num_elements + index] = T.max(a_int_ub[index], b_int_ub[index])
                    out_int_ub[num_elements * 2 + index] = T.truncmod(a_int_ub[index], b_int_ub[index])
                    out_float_ub[index] = T.min(a_float_ub[index], b_float_ub[index])
                    out_float_ub[num_elements + index] = T.max(a_float_ub[index], b_float_ub[index])
            T.copy(out_int_ub, out_int)
            T.copy(out_float_ub, out_float)

    return main


def _vector_shuffle_constructor_kernel():
    @T.prim_func
    def main(
        a_float: T.Tensor((2,), T.float32),
        a_int: T.Tensor((2,), T.int32),
        a_half: T.Tensor((2,), T.float16),
        a_bfloat: T.Tensor((2,), T.bfloat16),
        out_float: T.Tensor((2,), T.float32),
        out_int: T.Tensor((2,), T.int32),
        out_half: T.Tensor((2,), T.float16),
        out_bfloat: T.Tensor((2,), T.bfloat16),
    ):
        with T.Kernel(1):
            a_float_ub = T.alloc_shared((2,), T.float32)
            a_int_ub = T.alloc_shared((2,), T.int32)
            a_half_ub = T.alloc_shared((2,), T.float16)
            a_bfloat_ub = T.alloc_shared((2,), T.bfloat16)
            out_float_ub = T.alloc_shared((2,), T.float32)
            out_int_ub = T.alloc_shared((2,), T.int32)
            out_half_ub = T.alloc_shared((2,), T.float16)
            out_bfloat_ub = T.alloc_shared((2,), T.bfloat16)
            T.copy(a_float, a_float_ub)
            T.copy(a_int, a_int_ub)
            T.copy(a_half, a_half_ub)
            T.copy(a_bfloat, a_bfloat_ub)
            with T.SimtVF(threads=1):
                value_float = T.alloc_local((1,), "float32x2")
                value_int = T.alloc_local((1,), "int32x2")
                value_half = T.alloc_local((1,), "float16x2")
                value_bfloat = T.alloc_local((1,), "bfloat16x2")
                value_float[0] = T.float32x2(a_float_ub[0], a_float_ub[1])
                value_int[0] = T.int32x2(a_int_ub[0], a_int_ub[1])
                value_half[0] = T.float16x2(a_half_ub[0], a_half_ub[1])
                value_bfloat[0] = T.bfloat16x2(a_bfloat_ub[0], a_bfloat_ub[1])
                out_float_ub[T.Ramp(0, 1, 2)] = value_float[0]
                out_int_ub[T.Ramp(0, 1, 2)] = value_int[0]
                out_half_ub[T.Ramp(0, 1, 2)] = value_half[0]
                out_bfloat_ub[T.Ramp(0, 1, 2)] = value_bfloat[0]
            T.copy(out_float_ub, out_float)
            T.copy(out_int_ub, out_int)
            T.copy(out_half_ub, out_half)
            T.copy(out_bfloat_ub, out_bfloat)

    return main


def _vector_shuffle_buffer_load_kernel():
    @T.prim_func
    def main(a: T.Tensor((2,), T.float32), out: T.Tensor((1,), T.float32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((2,), T.float32)
            out_ub = T.alloc_shared((1,), T.float32)
            T.copy(a, a_ub)
            with T.SimtVF(threads=1):
                out_ub[0] = T.extract_lane(a_ub[T.Ramp(0, 1, 2)], 0)
            T.copy(out_ub, out)

    return main


def _local_var_float16x8_kernel():
    lanes = 8

    @T.prim_func
    def main(out: T.Tensor((lanes,), T.float16)):
        with T.Kernel(1):
            out_ub = T.alloc_shared((lanes,), T.float16)
            with T.SimtVF(threads=1):
                value = T.alloc_var("float16x8")
                out_ub[T.Ramp(0, 1, lanes)] = value
            T.copy(out_ub, out)

    return main


def _broadcast_kernel(dtype):
    lanes = 8

    @T.prim_func
    def main(a: T.Tensor((1,), dtype), out: T.Tensor((lanes,), dtype)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((1,), dtype)
            out_ub = T.alloc_shared((lanes,), dtype)
            T.copy(a, a_ub)
            with T.SimtVF(threads=1):
                value = T.alloc_local((1,), f"{dtype}x8")
                value[0] = T.Broadcast(a_ub[0], lanes)
                out_ub[T.Ramp(0, 1, lanes)] = value[0]
            T.copy(out_ub, out)

    return main


def test_vector_min_max_and_mod_are_scalarized():
    source = _source(_vector_min_max_mod_kernel(), target="ascend")
    assert source.count(".x = min(") == 2
    assert source.count(".y = min(") == 2
    assert source.count(".x = max(") == 2
    assert source.count(".y = max(") == 2
    assert ".x%" in source
    assert ".y%" in source


def test_shuffle_constructors_are_materialized_per_lane():
    source = _source(
        _vector_shuffle_constructor_kernel(),
        target="ascend",
        pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
    )
    assert "value_float[0] = float2(" not in source
    assert "value_int[0] = int2(" not in source
    assert "value_half[0] = uint1(" not in source
    assert "value_bfloat[0] = uint1(" not in source
    assert "((half2*)(&(" in source
    assert "((bfloat16x2_t*)(&(" in source


def test_shuffle_buffer_load_is_materialized_before_lane_access():
    source = _source(
        _vector_shuffle_buffer_load_kernel(),
        target="ascend",
        pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
    )
    assert re.search(r"float2\s+\w+\s*=\s*\*\(__ubuf__ float2\*\)", source)
    assert re.search(r"=\s*\w+\.x;", source)
    assert not re.search(r"\*\(__ubuf__ float2\*\)\([^;]*\)\.x", source)


def test_local_var_vector_initializer_is_emitted_before_declaration():
    source = _source(
        _local_var_float16x8_kernel(),
        target="ascend",
        pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
    )
    value_declaration = re.search(r"uint4 value = (\w+);", source)
    assert value_declaration is not None
    initializer_store = source.find("((half2*)(&(" + value_declaration.group(1))
    assert initializer_store != -1
    assert initializer_store < value_declaration.start()


@pytest.mark.parametrize("dtype,carrier", [("bfloat16", "bfloat16x2_t"), ("float16", "half2")])
def test_broadcast_is_materialized_per_lane(dtype, carrier):
    source = _source(_broadcast_kernel(dtype), target="ascend", pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True})
    assert f"(({carrier}*)(&(" in source
    assert ".x =" in source and ".y =" in source


@tilelang.testing.requires_ascend
def test_int64_strided_vector_load():
    import torch

    stride = T.dynamic("stride", dtype="int64")

    @T.prim_func
    def kernel(A: T.StridedTensor((2, 128), (stride, 1), "float32"), O: T.Tensor((2, 128), "float32")):
        T.assume(stride % 256 == 0)
        with T.Kernel(1), T.SimtVF(threads=64):
            values = T.alloc_fragment((128,), "float32")
            for j in T.Parallel(128):
                values[j] = A[1, j]
            for j in T.Parallel(128):
                O[0, j] = values[j]

    storage = torch.arange(512, dtype=torch.float32).reshape(2, 256).npu()
    out = torch.full((2, 128), -1.0, device="npu")
    tilelang.compile(kernel, target="ascend")(storage[:, :128], out)
    expected = torch.full((2, 128), -1.0)
    expected[0] = torch.arange(256, 384, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
