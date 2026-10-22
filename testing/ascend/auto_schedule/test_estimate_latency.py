"""Estimate latency contracts."""

import pytest
import tilelang
import tilelang.ascend.transform as ascend_transform
import tilelang.ascend.language as T
from tilelang import tvm
from testing.ascend.auto_schedule._task_utils import (
    _bind_target,
    _collect_schedule_units,
    _collect_task_metadata,
    _make_program,
    _materialize_schedule_units,
)


def _make_gemm_cost_program(m, n, k, dtype="float32", hf32=True, dynamic_k=False, l0=True, blockscaled=False):
    @T.prim_func
    def main(tail_k: T.int32):
        with T.Kernel(1):
            a = T.alloc_l0a((64, 64), dtype) if l0 else T.alloc_l1((64, 64), dtype)
            b = T.alloc_l0b((64, 64), dtype) if l0 else T.alloc_l1((64, 64), dtype)
            c = T.alloc_l0c((64, 64), "float32")
            if dtype == "float32":
                T.set_hf32_mode("nearest_even" if hf32 else None)
            effective_k = T.max(T.min(tail_k, k), 0) if dynamic_k else k
            if blockscaled:
                sfa = T.alloc_l0a_sf(a)
                sfb = T.alloc_l0b_sf(b)
                T.gemm_blockscaled(a[:m, :effective_k], b[:n, :effective_k], c[:m, :n], sfa, sfb, transpose_B=True, clear_accum=True)
            else:
                T.gemm(a[:m, :effective_k], b[:n, :effective_k], c[:m, :n], transpose_B=True, clear_accum=True)

    return main


def _estimate_gemm_cost(*args, **kwargs):
    mod = _bind_target(_make_gemm_cost_program(*args, **kwargs))
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = ascend_transform.EstimateLatency()(_materialize_schedule_units(mod))
    metadata = _collect_task_metadata(mod)
    return metadata["latency"][-1], metadata["ii"][-1]


@pytest.mark.parametrize("dynamic_k", [False, True])
def test_l0_gemm_effective_k_cost(dynamic_k):
    # A5 HF32: 4096 ops/cycle plus67 completion cycles. Allocation stays64.
    assert _estimate_gemm_cost(64, 64, 24, dynamic_k=dynamic_k) == (115, 48)
    assert _estimate_gemm_cost(64, 64, 32) == (131, 64)


def test_l0_gemm_mn_cost_rounds_to_fractals():
    assert _estimate_gemm_cost(24, 17, 32) == _estimate_gemm_cost(32, 32, 32)


@pytest.mark.parametrize("dtype,k,rounded_k", [("float32", 17, 24), ("bfloat16", 17, 32), ("float8_e4m3fn", 33, 64)])
def test_l0_gemm_k_cost_uses_compute_groups(dtype, k, rounded_k):
    assert _estimate_gemm_cost(64, 64, k, dtype=dtype) == _estimate_gemm_cost(64, 64, rounded_k, dtype=dtype)


def test_l0_gemm_cost_respects_fp32_mode():
    assert _estimate_gemm_cost(64, 64, 24, hf32=False) == (451, 384)


def test_l1_gemm_cost_uses_compute_groups():
    assert _estimate_gemm_cost(24, 24, 32, dtype="bfloat16", l0=False) == _estimate_gemm_cost(32, 32, 32, dtype="bfloat16", l0=False)


@pytest.mark.parametrize(
    "mn,dtype,measured",
    [
        (16, "float8_e4m3fn", (31, 6)),
        (64, "float8_e4m3fn", (57, 32)),
        (16, "float4_e2m1fn", (28, 3)),
        (64, "float4_e2m1fn", (41, 16)),
    ],
)
def test_blockscaled_gemm_measured_compute_cost(mn, dtype, measured):
    # CANN 9.2 / npusim Ascend950: steady latency/II with resident data/SF
    # and independent L0C outputs, covering small-tile and peak-throughput costs.
    assert _estimate_gemm_cost(mn, mn, 64, dtype=dtype, blockscaled=True) == measured


def test_dense_fp8_gemm_keeps_its_cost_model():
    assert _estimate_gemm_cost(16, 16, 64, dtype="float8_e4m3fn") == (10, 2)


def _make_dual_copy_program(split_n: bool):
    full_shape = (64, 256) if split_n else (128, 128)
    half_shape = (64, 128)

    @T.prim_func
    def main(
        A: T.Tensor(full_shape, "bfloat16"),
        B: T.Tensor(full_shape, "bfloat16"),
    ):
        with T.Kernel(1):
            temp = T.alloc_shared(half_shape, "bfloat16")
            T.dual_copy(A, temp)
            T.dual_copy(temp, B)

    return main


def _make_ordinary_copy_program(split_n: bool, use_sid: bool, bind_sid: bool = False):
    full_shape = (64, 256) if split_n else (128, 128)
    half_shape = (64, 128)

    @T.prim_func
    def main(
        A: T.Tensor(full_shape, "bfloat16"),
        B: T.Tensor(full_shape, "bfloat16"),
    ):
        with T.MixedKernel(1) as (_, sid):
            temp = T.alloc_shared(half_shape, "bfloat16")
            partition = T.bind(sid) if bind_sid else sid
            if not use_sid:
                partition = 0
            if split_n:
                T.copy(A[:, partition * 128 : (partition + 1) * 128], temp)
                T.copy(temp, B[:, partition * 128 : (partition + 1) * 128])
            else:
                T.copy(A[partition * 64 : (partition + 1) * 64, :], temp)
                T.copy(temp, B[partition * 64 : (partition + 1) * 64, :])

    return main


def _make_pure_vector_copy_program(split_n: bool):
    full_shape = (64, 256) if split_n else (128, 128)
    half_shape = (64, 128)

    @T.prim_func
    def main(
        A: T.Tensor(full_shape, "bfloat16"),
        B: T.Tensor(full_shape, "bfloat16"),
    ):
        with T.Kernel(64):
            temp = T.alloc_shared(half_shape, "bfloat16")
            if split_n:
                T.copy(A[:, :128], temp)
                T.copy(temp, B[:, :128])
            else:
                T.copy(A[:64, :], temp)
                T.copy(temp, B[:64, :])

    return main


def _make_mixed_aiv_copy_program(rows_per_aiv: int, columns: int):
    full_shape = (rows_per_aiv * 2, columns)
    half_shape = (rows_per_aiv, columns)

    @T.prim_func
    def main(
        A: T.Tensor(full_shape, "bfloat16"),
        B: T.Tensor(full_shape, "bfloat16"),
    ):
        with T.MixedKernel(1) as (_, sid):
            temp = T.alloc_shared(half_shape, "bfloat16")
            T.copy(A[sid * rows_per_aiv : (sid + 1) * rows_per_aiv, :], temp)
            T.copy(temp, B[sid * rows_per_aiv : (sid + 1) * rows_per_aiv, :])

    return main


def _make_symbolic_stride_copy_program():
    row_stride = T.dynamic("row_stride")

    @T.prim_func
    def main(
        A: T.StridedTensor(
            shape=[128, 128],
            strides=[row_stride, 1],
            dtype="bfloat16",
        ),
        B: T.StridedTensor(
            shape=[128, 128],
            strides=[row_stride, 1],
            dtype="bfloat16",
        ),
    ):
        with T.MixedKernel(1):
            temp = T.alloc_shared((64, 128), "bfloat16")
            T.copy(A[:64, :], temp)
            T.copy(temp, B[:64, :])

    return main


def _make_high_dim_strided_copy_program():
    @T.prim_func
    def main(
        A: T.Tensor((4, 2, 128), "bfloat16"),
        B: T.Tensor((4, 2, 128), "bfloat16"),
    ):
        with T.MixedKernel(1):
            temp = T.alloc_shared((2, 1, 128), "bfloat16")
            T.copy(A[0:2, 0:1, :], temp)
            T.copy(temp, B[0:2, 0:1, :])

    return main


def _make_rank_one_dual_copy_program():
    @T.prim_func
    def main(
        A: T.Tensor((8192,), "bfloat16"),
        B: T.Tensor((8192,), "bfloat16"),
    ):
        with T.Kernel(1):
            temp = T.alloc_shared((4096,), "bfloat16")
            T.dual_copy(A, temp)
            T.dual_copy(temp, B)

    return main


def _make_n512_dual_copy_program():
    @T.prim_func
    def main(
        A: T.Tensor((32, 512), "bfloat16"),
        B: T.Tensor((32, 512), "bfloat16"),
    ):
        with T.Kernel(1):
            temp = T.alloc_shared((32, 256), "bfloat16")
            T.dual_copy(A, temp)
            T.dual_copy(temp, B)

    return main


def _make_rank_one_strided_copy_program(explicit_rows: bool):
    if explicit_rows:

        @T.prim_func
        def main(A: T.StridedTensor([4096, 1], [2, 1], "bfloat16")):
            with T.Kernel(64):
                temp = T.alloc_shared((4096, 1), "bfloat16")
                T.copy(A[:, :], temp)

    else:

        @T.prim_func
        def main(A: T.StridedTensor([4096], [2], "bfloat16")):
            with T.Kernel(64):
                temp = T.alloc_shared((4096,), "bfloat16")
                T.copy(A[:], temp)

    return main


def _make_sid_padded_copy_program():
    @T.prim_func
    def main(A: T.StridedTensor([128, 32], [64, 1], "bfloat16")):
        with T.MixedKernel(1) as (_, sid):
            temp = T.alloc_shared((64, 32), "bfloat16")
            T.copy(A[sid * 64 : (sid + 1) * 64, :], temp)

    return main


def _make_single_sid_copy_program():
    @T.prim_func
    def main(A: T.Tensor((64, 128), "bfloat16")):
        with T.MixedKernel(1, sids=1) as (_, sid):
            temp = T.alloc_shared((64, 128), "bfloat16")
            T.copy(A[sid * 64 : (sid + 1) * 64, :], temp)

    return main


def _make_gm_to_l1_program(size: int):
    @T.prim_func
    def main(A: T.Tensor((size, size), "bfloat16")):
        with T.Kernel(1):
            temp = T.alloc_l1((size, size), "bfloat16")
            T.copy(A, temp)

    return main


def _make_two_small_gm_to_l1_copies_program():
    @T.prim_func
    def main(
        A: T.Tensor((16, 16), "bfloat16"),
        B: T.Tensor((16, 16), "bfloat16"),
    ):
        with T.Kernel(1):
            a_l1 = T.alloc_l1((16, 16), "bfloat16")
            b_l1 = T.alloc_l1((16, 16), "bfloat16")
            with T.Task():
                T.copy(A, a_l1)
                T.copy(B, b_l1)

    return main


def _make_l0c_copy_program(dst_scope: str, dtype: str):
    if dst_scope == "ub":

        @T.prim_func
        def main():
            with T.Kernel(1):
                src = T.alloc_l0c((128, 128), "float32")
                dst = T.alloc_shared((128, 128), dtype)
                T.copy(src, dst)

    else:

        @T.prim_func
        def main(dst: T.Tensor((128, 128), dtype)):
            with T.Kernel(1):
                src = T.alloc_l0c((128, 128), "float32")
                T.copy(src, dst)

    return main


def _make_narrow_fixpipe_dual_program(split_n: bool):
    full_shape = (16, 32) if split_n else (32, 16)

    @T.prim_func
    def main():
        with T.Kernel(1):
            src = T.alloc_l0c(full_shape, "float32")
            dst = T.alloc_shared((16, 16), "float32")
            T.dual_copy(src, dst)

    return main


def _make_unit_axis_fixpipe_dual_program():
    @T.prim_func
    def main():
        with T.Kernel(1):
            src = T.alloc_l0c((1, 32, 16), "float32")
            dst = T.alloc_shared((16, 16, 1), "float32")
            T.dual_copy(src, dst)

    return main


def _estimate_copy_metadata(program, rewrite_dual_copy=False):
    mod = _bind_target(program)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    if rewrite_dual_copy:
        mod = ascend_transform.RewriteDualCopy()(mod)
    mod = _materialize_schedule_units(mod)
    mod = ascend_transform.EstimateLatency()(mod)
    metadata = _collect_task_metadata(mod)
    return list(zip(metadata["latency"], metadata["ii"]))


def test_estimate_latency_annotates_all_materialized_tasks():
    mod = ascend_transform.EstimateLatency()(_materialize_schedule_units(_bind_target(_make_program())))
    metadata = _collect_task_metadata(mod)

    assert len(metadata["latency"]) == 2
    assert len(metadata["ii"]) == 2
    assert all(latency >= 0 for latency in metadata["latency"])
    assert all(ii > 0 for ii in metadata["ii"])
    assert _collect_schedule_units(mod)


def test_frontend_task_hints_feed_shared_task_metadata():
    mod = ascend_transform.EstimateLatency()(_materialize_schedule_units(_bind_target(_make_program(latency=17, ii=5))))
    metadata = _collect_task_metadata(mod)

    assert 17 in metadata["latency"]
    assert 5 in metadata["ii"]


@pytest.mark.parametrize("split_n", [False, True])
def test_rewritten_dual_copy_matches_equivalent_aiv_mte_access(split_n):
    unrewritten = _estimate_copy_metadata(_make_dual_copy_program(split_n))
    rewritten = _estimate_copy_metadata(_make_dual_copy_program(split_n), rewrite_dual_copy=True)
    sid_partitioned = _estimate_copy_metadata(_make_ordinary_copy_program(split_n, use_sid=True))
    bound_sid_partitioned = _estimate_copy_metadata(_make_ordinary_copy_program(split_n, use_sid=True, bind_sid=True))
    fixed_partition = _estimate_copy_metadata(_make_ordinary_copy_program(split_n, use_sid=False))
    pure_vector = _estimate_copy_metadata(_make_pure_vector_copy_program(split_n))

    expected = [(403, 328), (478, 298)] if split_n else [(418, 328), (465, 285)]
    assert unrewritten == rewritten == sid_partitioned == fixed_partition == pure_vector == expected
    assert bound_sid_partitioned == [(1, 1), *expected]


def test_aiv_mte_large_hbm_uses_full_occupancy_ii():
    estimated = _estimate_copy_metadata(_make_mixed_aiv_copy_program(64, 128))

    assert estimated == [(418, 328), (465, 285)]


def test_aiv_mte_small_copy_keeps_descriptor_floor_ii():
    estimated = _estimate_copy_metadata(_make_mixed_aiv_copy_program(1, 64))

    assert estimated == [(93, 13), (183, 10)]


def test_cthread_symbolic_stride_uses_conservative_unknown_split():
    estimated = _estimate_copy_metadata(_make_symbolic_stride_copy_program())

    assert estimated == [(403, 328), (478, 298)]


def test_aiv_mte_high_dim_copy_uses_actual_mte_row_axis():
    estimated = _estimate_copy_metadata(_make_high_dim_strided_copy_program())

    assert estimated == [(86, 13), (190, 10)]


def test_rank_one_dual_copy_keeps_geometry_across_rewrite():
    program = _make_rank_one_dual_copy_program()

    assert (
        _estimate_copy_metadata(program)
        == _estimate_copy_metadata(program, rewrite_dual_copy=True)
        == [
            (254, 164),
            (323, 143),
        ]
    )


def test_n512_copy_uses_measured_full_width_mte3_rate():
    program = _make_n512_dual_copy_program()

    assert (
        _estimate_copy_metadata(program)
        == _estimate_copy_metadata(program, rewrite_dual_copy=True)
        == [
            (403, 328),
            (465, 285),
        ]
    )


def test_equivalent_strided_regions_share_mte_row_geometry():
    rank_one = _estimate_copy_metadata(_make_rank_one_strided_copy_program(explicit_rows=False))
    explicit_rows = _estimate_copy_metadata(_make_rank_one_strided_copy_program(explicit_rows=True))

    assert rank_one == explicit_rows == [(403, 328)]


def test_sid_partition_does_not_override_strided_mte_geometry():
    estimated = _estimate_copy_metadata(_make_sid_padded_copy_program())

    assert estimated == [(239, 164)]


def test_single_sid_copy_cost():
    assert _estimate_copy_metadata(_make_single_sid_copy_program()) == [(418, 328)]


def test_updated_non_aiv_copy_costs_match_full_path_remeasurement():
    assert _estimate_copy_metadata(_make_gm_to_l1_program(16)) == [(196, 64)]
    assert _estimate_copy_metadata(_make_gm_to_l1_program(128)) == [(518, 328)]
    assert _estimate_copy_metadata(_make_two_small_gm_to_l1_copies_program()) == [(260, 128)]
    assert _estimate_copy_metadata(_make_l0c_copy_program("ub", "float32")) == [(570, 514)]
    assert _estimate_copy_metadata(_make_l0c_copy_program("gm", "float32")) == [(712, 514)]


@pytest.mark.parametrize(
    "rows,k,sf_dtype,dynamic_rows,measured",
    [
        (16, 64, "uint16", False, (29, 2)),
        (128, 128, "int16", False, (44, 17)),
        (256, 128, "uint8", False, (60, 33)),
        (128, 128, "int16", True, (44, 17)),
    ],
)
def test_mx_sf_copy_cost_uses_region_payload(rows, k, sf_dtype, dynamic_rows, measured):
    # CANN 9.2 / npusim: 32, 512 and 1024 B SF loads on either L0 side.
    # Slice a larger allocation so the cost must use the region and SF dtype.
    sf_columns = k // (32 * (tvm.DataType(sf_dtype).bits // 8))

    @T.prim_func
    def main(tail_rows: T.int32):
        with T.Kernel(1):
            src = T.alloc_l1((256, sf_columns * 2), sf_dtype)
            a = T.alloc_l0a((256, k), "float8_e4m3fn")
            b = T.alloc_l0b((256, k), "float8_e4m3fn")
            sfa = T.alloc_l0a_sf(a, sf_dtype=sf_dtype)
            sfb = T.alloc_l0b_sf(b, sf_dtype=sf_dtype)
            copy_rows = T.max(T.min(tail_rows, rows), 0) if dynamic_rows else rows
            T.copy(src[:copy_rows, sf_columns : sf_columns * 2], sfa[:copy_rows, :])
            T.copy(src[:copy_rows, sf_columns : sf_columns * 2], sfb[:copy_rows, :])

    assert _estimate_copy_metadata(main)[-2:] == [measured, measured]


def test_fixpipe_quant_cost_uses_destination_payload_bytes():
    assert _estimate_copy_metadata(_make_l0c_copy_program("ub", "bfloat16")) == [(314, 258)]
    assert _estimate_copy_metadata(_make_l0c_copy_program("gm", "bfloat16")) == [(456, 258)]


@pytest.mark.parametrize("split_n", [False, True])
def test_narrow_fixpipe_dual_uses_physical_trailing_row_width(split_n):
    assert _estimate_copy_metadata(_make_narrow_fixpipe_dual_program(split_n)) == [(71, 16)]


def test_fixpipe_dual_ignores_destination_unit_axis_for_row_width():
    assert _estimate_copy_metadata(_make_unit_axis_fixpipe_dual_program()) == [(71, 16)]


def test_task_cost_arguments_are_validated():
    with pytest.raises(ValueError, match="non-negative integer"):
        _make_program(latency=-1)
    with pytest.raises(ValueError, match="positive integer"):
        _make_program(ii=0)
    with pytest.raises(ValueError, match="greater than or equal to ii"):
        _make_program(latency=1, ii=2)
    with pytest.raises(tvm.error.InternalError, match="greater than or equal to ii"):
        ascend_transform.EstimateLatency()(_materialize_schedule_units(_bind_target(_make_program(latency=0))))


if __name__ == "__main__":
    pytest.main([__file__])
