"""Regression tests for small-vector expression codegen in the PTO path."""

import re

import pytest

import tilelang
import tilelang.testing
import tilelang.ascend.language as T


def _source(func, *, target, pass_configs=None):
    with tilelang.tvm.target.Target(target), tilelang.transform.PassContext(config=pass_configs or {}):
        return tilelang.lower(func, target=target).kernel_source


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


@pytest.mark.pto
@pytest.mark.parametrize("explicit", [False, True], ids=["parallel-index", "strided-ramp"])
def test_pto_ramp_value_is_one_vector_argument(explicit):
    source = _source(_ramp_value_kernel(explicit), target={"kind": "ascend", "keys": ["pto", "ascend"]})
    lanes = 4 if explicit else 2
    assert re.search(rf"pto\.store\(.*tl\.vector_from_list\(pto\.si32, .*contiguous={lanes}\)", source)


@pytest.mark.pto
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


@pytest.mark.pto
@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["pto"])
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


@pytest.mark.pto
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


@pytest.mark.pto
@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["pto"])
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


@pytest.mark.pto
def test_pto_vector_local_cross_slot_offsets():
    source = tilelang.lower(_vector_local_cross_slot_kernel(), target="pto", enable_device_compile=False).kernel_source
    for offset in (2, 4):
        assert f"pto.load(vals, {offset}, contiguous=4)" in source
    assert re.search(r"pto\.store\([^\n]+, vals, 2, contiguous=4\)", source)


@tilelang.testing.requires_ascend
@pytest.mark.pto
def test_pto_vector_local_cross_slot_mutation_preserves_neighbors():
    import torch

    a = torch.arange(1, 13, dtype=torch.float32, device="npu")
    kernel = tilelang.compile(_vector_local_cross_slot_kernel(), target="pto", out_idx=-1)
    result = kernel(a)
    torch.npu.synchronize()
    expected = torch.cat((a[2:6], a[4:8], a[:2], a[8:12], a[6:8]))
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.pto
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


@pytest.mark.pto
@pytest.mark.parametrize("vector_elements", [False, True], ids=["scalar-elements", "vector-elements"])
def test_pto_local_vector_load_restores_unsigned_type(vector_elements):
    source = tilelang.lower(_unsigned_local_min_max_kernel(vector_elements), target="pto", enable_device_compile=False).kernel_source
    assert "pto.alloc_buffer((4,), pto.i32)" in source
    for offset in (0, 2):
        assert f"pto.cast(pto.load(vals, {offset}, contiguous=2), pto.ui32)" in source


@pytest.mark.pto
@tilelang.testing.requires_ascend
@pytest.mark.parametrize("target", ["pto"])
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
