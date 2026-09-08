import re

import pytest
import tilelang.language as T
import tilelang.testing
import tvm

from tilelang.engine.lower import lower


TILE = 256


def _generated_cross_core_slots(source: str) -> set[int]:
    return {int(value) % 16 for value in re.findall(r"CrossCore(?:Set|Wait)Flag<4, [^>]+>\((\d+)\)", source)}


def _make_sparse_explicit_cross_core_flag_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        C: T.Tensor((TILE, TILE), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            with T.PerCoreTask():
                T.dual_copy(accum, temp)
                T.ascend_sync_inter_arrive("PIPE_FIX", 15)
                T.ascend_sync_inter_wait("PIPE_FIX", 15)
            T.dual_copy(temp, C)

    return main


def test_sparse_explicit_cross_core_flag_allocation():
    source = lower(_make_sparse_explicit_cross_core_flag_program(), target="ascend").kernel_source
    assert "AscendC::CrossCoreSetFlag<0, PIPE_FIX>(15);" in source
    assert _generated_cross_core_slots(source) == {0}


def _make_bounded_dynamic_cross_core_flag_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        C: T.Tensor((TILE, TILE), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.dual_copy(accum, temp)
            for outer in T.Serial(2):
                for inner in T.Serial(2):
                    flag_id = T.bind(outer * 2 + inner)
                    T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
                    T.ascend_sync_inter_wait("PIPE_FIX", flag_id)
            T.dual_copy(temp, C)

    return main


def test_bounded_dynamic_cross_core_flag_allocation():
    with tvm.transform.PassContext(config={"tl.Simplify": {"enable_simplify_let_inline": False}}):
        source = lower(_make_bounded_dynamic_cross_core_flag_program(), target="ascend").kernel_source
    assert _generated_cross_core_slots(source) == {4}


def _make_guarded_dynamic_cross_core_flag_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        C: T.Tensor((TILE, TILE), "float32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.dual_copy(accum, temp)
            for flag_id in T.Serial(32):
                if flag_id < 4:
                    T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
                    T.ascend_sync_inter_wait("PIPE_FIX", flag_id)
            T.dual_copy(temp, C)

    return main


def test_guarded_dynamic_cross_core_flag_allocation():
    source = lower(_make_guarded_dynamic_cross_core_flag_program(), target="ascend").kernel_source
    assert _generated_cross_core_slots(source) == {4}


def _make_assumed_dynamic_cross_core_flag_program():
    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        C: T.Tensor((TILE, TILE), "float32"),
        flag_id: T.int32,
    ):
        with T.Kernel(1):
            T.assume(flag_id >= 0)
            T.assume(flag_id < 4)
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            T.dual_copy(accum, temp)
            T.ascend_sync_inter_arrive("PIPE_FIX", flag_id)
            T.ascend_sync_inter_wait("PIPE_FIX", flag_id)
            T.dual_copy(temp, C)

    return main


def test_assumed_dynamic_cross_core_flag_allocation():
    source = lower(_make_assumed_dynamic_cross_core_flag_program(), target="ascend").kernel_source
    assert _generated_cross_core_slots(source) == {4}


def _make_unbounded_dynamic_cross_core_flag_program():
    flag_extent = T.dynamic("flag_extent")

    @T.prim_func
    def main(
        A: T.Tensor((TILE, TILE), "bfloat16"),
        B: T.Tensor((TILE, TILE), "bfloat16"),
        C: T.Tensor((TILE, TILE), "float32"),
        Marker: T.Tensor((flag_extent,), "int32"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            b_l1 = T.alloc_l1((TILE, TILE), "bfloat16")
            accum = T.alloc_l0c((TILE, TILE), "float32")
            temp = T.alloc_shared((TILE // 2, TILE), "float32")
            T.copy(A, a_l1)
            T.copy(B, b_l1)
            T.gemm(a_l1, b_l1, accum, transpose_B=True, clear_accum=True)
            with T.PerCoreTask():
                T.dual_copy(accum, temp)
                T.ascend_sync_inter_arrive("PIPE_FIX", flag_extent)
                T.ascend_sync_inter_wait("PIPE_FIX", flag_extent)
            T.dual_copy(temp, C)

    return main


def test_unbounded_dynamic_cross_core_flag_is_rejected():
    with pytest.raises(tvm.error.InternalError, match="integer range cannot be bounded"):
        lower(_make_unbounded_dynamic_cross_core_flag_program(), target="ascend")


if __name__ == "__main__":
    tilelang.testing.main()
