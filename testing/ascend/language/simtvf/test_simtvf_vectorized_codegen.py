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


def _ramp_value_kernel(explicit):
    num_elements = 128 if explicit else 256

    @T.prim_func
    def main(out: T.Tensor((num_elements,), T.int32)):
        with T.Kernel(1):
            out_ub = T.alloc_shared((num_elements,), T.int32)
            with T.SimtVF(threads=32):
                if explicit:
                    lane = T.get_thread_binding()
                    out_ub[T.Ramp(lane * 4, 1, 4)] = T.Ramp(lane * 12 + 7, 3, 4)
                else:
                    indices = T.alloc_fragment((num_elements,), T.int32)
                    for i in T.Parallel(num_elements):
                        indices[i] = i
                    for i in T.Parallel(num_elements):
                        out_ub[i] = indices[i]
            T.copy(out_ub, out)

    return main


@pytest.mark.parametrize("explicit", [False, True], ids=["parallel-index", "strided-ramp"])
def test_pto_ramp_value_is_one_vector_argument(explicit):
    source = _source(_ramp_value_kernel(explicit), target={"kind": "ascend", "keys": ["pto", "ascend"]})
    lanes = 4 if explicit else 2
    assert re.search(rf"pto\.store\(.*tl\.vector_from_list\(pto\.i32, .*contiguous={lanes}\)", source)


def test_pto_ramp_operands_are_evaluated_once():
    @T.prim_func
    def main():
        with T.Kernel(1), T.SimtVF(threads=1):
            T.evaluate(
                T.call_extern(
                    "int32",
                    "consume",
                    T.Ramp(
                        T.call_extern("int32", "next_base"),
                        T.call_extern("int32", "next_stride"),
                        4,
                    ),
                )
            )

    source = _source(main, target={"kind": "ascend", "keys": ["pto", "ascend"]})
    for operand in ("next_base", "next_stride"):
        assert source.count(f"{operand}()") == 1


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("explicit", [False, True], ids=["parallel-index", "strided-ramp"])
def test_ramp_value_preserves_all_lanes(explicit, target):
    import torch

    kernel = tilelang.compile(_ramp_value_kernel(explicit), target=target, out_idx=-1)
    result = kernel()
    torch.npu.synchronize()
    expected = torch.arange(128 if explicit else 256, dtype=torch.int32)
    if explicit:
        expected = expected * 3 + 7
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


_VECTOR_LOCAL_CASES = [
    pytest.param(2, False, id="float2-literal"),
    pytest.param(2, True, id="float2-unrolled"),
    pytest.param(4, False, id="float4-literal"),
    pytest.param(4, True, id="float4-unrolled"),
]


def _vector_local_kernel(lanes, unrolled):
    @T.prim_func
    def main(a: T.Tensor((3 * lanes,), T.float32), out: T.Tensor((4 * lanes,), T.float32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((3 * lanes,), T.float32)
            out_ub = T.alloc_shared((4 * lanes,), T.float32)
            T.copy(a, a_ub)
            with T.SimtVF(threads=1):
                vals = T.alloc_local((2,), f"float32x{lanes}")
                if unrolled:
                    for t in T.unroll(2):
                        vals[t] = a_ub[T.Ramp(t * lanes, 1, lanes)]
                else:
                    vals[0] = a_ub[T.Ramp(0, 1, lanes)]
                    vals[1] = a_ub[T.Ramp(lanes, 1, lanes)]
                out_ub[T.Ramp(0, 1, lanes)] = vals[0]
                out_ub[T.Ramp(lanes, 1, lanes)] = vals[1]
                vals[0] = a_ub[T.Ramp(2 * lanes, 1, lanes)]
                out_ub[T.Ramp(2 * lanes, 1, lanes)] = vals[0]
                out_ub[T.Ramp(3 * lanes, 1, lanes)] = vals[1]
            T.copy(out_ub, out)

    return main


@pytest.mark.parametrize("lanes,unrolled", _VECTOR_LOCAL_CASES)
def test_pto_vector_local_uses_scalar_addressed_storage(lanes, unrolled):
    source = tilelang.lower(_vector_local_kernel(lanes, unrolled), target="pto", enable_device_compile=False).kernel_source
    assert f"vals = pto.alloc_buffer(({2 * lanes},), pto.f32)" in source
    assert "vals[" not in source
    for offset in (0, lanes):
        assert f"pto.load(vals, {offset}, contiguous={lanes})" in source
    assert re.search(rf"pto\.store\([^\n]+, vals, 0, contiguous={lanes}\)", source)
    offset = rf"t \* {lanes}" if unrolled else str(lanes)
    assert re.search(rf"pto\.store\([^\n]+, vals, {offset}, contiguous={lanes}\)", source)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("lanes,unrolled", _VECTOR_LOCAL_CASES)
def test_vector_local_slots_observe_mutation(lanes, unrolled, target):
    import torch

    a = torch.arange(1, 3 * lanes + 1, dtype=torch.float32, device="npu")
    kernel = tilelang.compile(_vector_local_kernel(lanes, unrolled), target=target, out_idx=-1)
    result = kernel(a)
    torch.npu.synchronize()
    expected = torch.cat((a[: 2 * lanes], a[2 * lanes :], a[lanes : 2 * lanes]))
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def _vector_local_cross_slot_kernel(load_stride=1, store_stride=1):
    @T.prim_func
    def main(a: T.Tensor((12,), T.float32), out: T.Tensor((16,), T.float32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((12,), T.float32)
            out_ub = T.alloc_shared((16,), T.float32)
            T.copy(a, a_ub)
            with T.SimtVF(threads=1):
                vals = T.alloc_local((4,), "float32x2")
                for i in T.unroll(4):
                    vals[i] = a_ub[T.Ramp(i * 2, 1, 2)]
                out_ub[T.Ramp(0, 1, 4)] = vals[T.Ramp(1, load_stride, 2)]
                out_ub[T.Ramp(4, 1, 4)] = vals[T.Ramp(2, 1, 2)]
                vals[T.Ramp(1, store_stride, 2)] = a_ub[T.Ramp(8, 1, 4)]
                for i in T.unroll(4):
                    out_ub[T.Ramp(8 + i * 2, 1, 2)] = vals[i]
            T.copy(out_ub, out)

    return main


def test_pto_vector_local_cross_slot_offsets():
    source = tilelang.lower(_vector_local_cross_slot_kernel(), target="pto", enable_device_compile=False).kernel_source
    for offset in (2, 4):
        assert f"pto.load(vals, {offset}, contiguous=4)" in source
    assert re.search(r"pto\.store\([^\n]+, vals, 2, contiguous=4\)", source)


@tilelang.testing.requires_ascend
def test_pto_vector_local_cross_slot_mutation_preserves_neighbors():
    import torch

    a = torch.arange(1, 13, dtype=torch.float32, device="npu")
    kernel = tilelang.compile(_vector_local_cross_slot_kernel(), target="pto", out_idx=-1)
    result = kernel(a)
    torch.npu.synchronize()
    expected = torch.cat((a[2:6], a[4:8], a[:2], a[8:12], a[6:8]))
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("access_kind", ["load", "store"])
def test_pto_vector_local_cross_slot_rejects_noncontiguous_ramp(access_kind):
    func = _vector_local_cross_slot_kernel(**{f"{access_kind}_stride": 2})
    with pytest.raises(tilelang.tvm.error.InternalError, match=rf"vector {access_kind}.*Ramp stride must be 1"):
        tilelang.lower(func, target="pto", enable_device_compile=False)


def _unsigned_local_min_max_kernel(vector_elements):
    shape = (2,) if vector_elements else (4,)
    dtype = "uint32x2" if vector_elements else "uint32"

    @T.prim_func
    def main(a: T.Tensor((4,), T.int32), out: T.Tensor((4,), T.int32)):
        with T.Kernel(1):
            a_ub = T.alloc_shared((4,), T.int32)
            out_ub = T.alloc_shared((4,), T.int32)
            T.copy(a, a_ub)
            with T.SimtVF(threads=1):
                vals = T.alloc_local(shape, dtype)
                if vector_elements:
                    vals[0] = T.Cast("uint32x2", a_ub[T.Ramp(0, 1, 2)])
                    vals[1] = T.Cast("uint32x2", a_ub[T.Ramp(2, 1, 2)])
                    out_ub[T.Ramp(0, 1, 2)] = T.Cast("int32x2", T.min(vals[0], vals[1]))
                    out_ub[T.Ramp(2, 1, 2)] = T.Cast("int32x2", T.max(vals[0], vals[1]))
                else:
                    vals[T.Ramp(0, 1, 2)] = T.Cast("uint32x2", a_ub[T.Ramp(0, 1, 2)])
                    vals[T.Ramp(2, 1, 2)] = T.Cast("uint32x2", a_ub[T.Ramp(2, 1, 2)])
                    out_ub[T.Ramp(0, 1, 2)] = T.Cast("int32x2", T.min(vals[T.Ramp(0, 1, 2)], vals[T.Ramp(2, 1, 2)]))
                    out_ub[T.Ramp(2, 1, 2)] = T.Cast("int32x2", T.max(vals[T.Ramp(0, 1, 2)], vals[T.Ramp(2, 1, 2)]))
            T.copy(out_ub, out)

    return main


@pytest.mark.parametrize("vector_elements", [False, True], ids=["scalar-elements", "vector-elements"])
def test_pto_local_vector_load_restores_unsigned_type(vector_elements):
    source = tilelang.lower(_unsigned_local_min_max_kernel(vector_elements), target="pto", enable_device_compile=False).kernel_source
    assert "pto.alloc_buffer((4,), pto.i32)" in source
    for offset in (0, 2):
        assert f"pto.cast(pto.load(vals, {offset}, contiguous=2), pto.ui32)" in source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["ascend", "pto"])
@pytest.mark.parametrize("vector_elements", [False, True], ids=["scalar-elements", "vector-elements"])
def test_unsigned_local_vector_min_max(vector_elements, target):
    import torch

    # Use int32 tensors to transport uint32 bit patterns without uint32 NPU ops.
    a = torch.tensor([-2147483648, -1, 1, 1], dtype=torch.int32, device="npu")
    kernel = tilelang.compile(_unsigned_local_min_max_kernel(vector_elements), target=target, out_idx=-1)
    result = kernel(a)
    torch.npu.synchronize()
    expected = torch.tensor([1, 1, -2147483648, -1], dtype=torch.int32)
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


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
