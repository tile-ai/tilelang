"""Regression tests for vectorized Select codegen in Ascend SIMT blocks."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


def _vector_condition_kernel():
    num_topk = 6

    @T.prim_func
    def main(out: T.Tensor((num_topk * 2,), T.int32)):
        with T.Kernel(1):
            out_ub = T.alloc_shared((num_topk * 2,), T.int32)
            with T.SimtVF(threads=32):
                lane = T.get_thread_binding()
                if lane < num_topk:
                    for i in T.vectorized(2):
                        out_ub[2 * lane + i] = T.Select(i == 0, lane + 10, -1)
            T.copy(out_ub, out)

    return main


def _broadcast_condition_kernel():
    num_threads = 32

    @T.prim_func
    def main(out: T.Tensor((num_threads * 2,), T.int32)):
        with T.Kernel(1):
            out_ub = T.alloc_shared((num_threads * 2,), T.int32)
            with T.SimtVF(threads=num_threads):
                lane = T.get_thread_binding()
                for i in T.vectorized(2):
                    out_ub[2 * lane + i] = T.Select(lane < 6, i, -1)
            T.copy(out_ub, out)

    return main


def _wide_vector_condition_kernel(dtype):
    num_elements = 16
    vector_lanes = 8

    @T.prim_func
    def main(
        lhs: T.Tensor((num_elements,), dtype),
        rhs: T.Tensor((num_elements,), dtype),
        out: T.Tensor((num_elements,), dtype),
    ):
        with T.Kernel(1):
            lhs_ub = T.alloc_shared((num_elements,), dtype)
            rhs_ub = T.alloc_shared((num_elements,), dtype)
            out_ub = T.alloc_shared((num_elements,), dtype)
            T.copy(lhs, lhs_ub)
            T.copy(rhs, rhs_ub)
            with T.SimtVF(threads=2):
                lane = T.get_thread_binding()
                indices = T.Ramp(vector_lanes * lane, 1, vector_lanes)
                out_ub[indices] = T.Select(lhs_ub[indices] > rhs_ub[indices], lhs_ub[indices], rhs_ub[indices])
            T.copy(out_ub, out)

    return main


def _wide_broadcast_condition_kernel(vector_lanes):
    @T.prim_func
    def main(
        lhs: T.Tensor((1,), T.float16),
        rhs: T.Tensor((1,), T.float16),
        out: T.Tensor((vector_lanes,), T.float16),
    ):
        with T.Kernel(1):
            lhs_ub = T.alloc_shared((1,), T.float16)
            rhs_ub = T.alloc_shared((1,), T.float16)
            out_ub = T.alloc_shared((vector_lanes,), T.float16)
            T.copy(lhs, lhs_ub)
            T.copy(rhs, rhs_ub)
            with T.SimtVF(threads=1):
                indices = T.Ramp(0, 1, vector_lanes)
                condition = T.Broadcast(lhs_ub[0] > rhs_ub[0], vector_lanes)
                lhs_value = T.Broadcast(lhs_ub[0], vector_lanes)
                rhs_value = T.Broadcast(rhs_ub[0], vector_lanes)
                out_ub[indices] = T.Select(condition, lhs_value, rhs_value)
            T.copy(out_ub, out)

    return main


def _negated_vector_conditions_kernel():
    vector_lanes = 8

    @T.prim_func
    def main(
        lhs: T.Tensor((1,), T.float16),
        rhs: T.Tensor((1,), T.float16),
        out: T.Tensor((vector_lanes * 2,), T.float16),
    ):
        with T.Kernel(1):
            lhs_ub = T.alloc_shared((1,), T.float16)
            rhs_ub = T.alloc_shared((1,), T.float16)
            out_ub = T.alloc_shared((vector_lanes * 2,), T.float16)
            T.copy(lhs, lhs_ub)
            T.copy(rhs, rhs_ub)
            with T.SimtVF(threads=1):
                left = T.Broadcast(lhs_ub[0] > rhs_ub[0], vector_lanes)
                right = T.Broadcast(rhs_ub[0] > lhs_ub[0], vector_lanes)
                lhs_value = T.Broadcast(lhs_ub[0], vector_lanes)
                rhs_value = T.Broadcast(rhs_ub[0], vector_lanes)
                out_ub[T.Ramp(0, 1, vector_lanes)] = T.Select(~left, lhs_value, rhs_value)
                out_ub[T.Ramp(vector_lanes, 1, vector_lanes)] = T.Select(not (left ^ right), lhs_value, rhs_value)
            T.copy(out_ub, out)

    return main


def test_vectorized_select_with_vector_condition():
    kernel = tilelang.compile(_vector_condition_kernel(), target="ascend")
    source = kernel.get_kernel_source()
    assert "ushort2" in source
    assert "make_int2((0)+(1*0), (0)+(1*1))" in source
    assert ".x = (bool(" in source
    assert ".y = (bool(" in source


def test_vectorized_select_simplifies_loop_invariant_condition():
    kernel = tilelang.compile(_broadcast_condition_kernel(), target="ascend")
    source = kernel.get_kernel_source()
    assert "make_ushort2" not in source
    assert ") ? make_int2" in source


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_wide_vectorized_select_uses_packed_predicate_carrier(dtype):
    kernel = tilelang.compile(
        _wide_vector_condition_kernel(dtype),
        target="ascend",
        pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
    )
    source = kernel.get_kernel_source()
    assert "uint4" in source
    assert source.count("((ushort2*)(&(") >= 16
    assert "bool(((ushort2*)(&(" in source


@pytest.mark.parametrize("vector_lanes", [6, 8])
def test_wide_broadcast_condition_is_materialized_per_lane(vector_lanes):
    kernel = tilelang.compile(
        _wide_broadcast_condition_kernel(vector_lanes),
        target="ascend",
        pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
    )
    source = kernel.get_kernel_source()
    carrier = f"uint{vector_lanes // 2}"
    assert carrier in source
    assert f"make_{carrier}(" not in source
    assert source.count("((ushort2*)(&(") >= vector_lanes * 2
    assert "bool(((ushort2*)(&(" in source


def test_vector_predicate_not_is_materialized_per_lane():
    kernel = tilelang.compile(
        _negated_vector_conditions_kernel(),
        target="ascend",
        pass_configs={tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True},
    )
    source = kernel.get_kernel_source()
    assert "(~" not in source
    assert source.count(" = (!") >= 16


if __name__ == "__main__":
    tilelang.testing.main()
