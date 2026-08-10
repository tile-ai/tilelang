import tilelang
import tilelang.language as T
import tilelang.testing
import pytest
import torch


_COMPILE_FLAGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

_PROJECTED_ROW_LOOP_LAYOUT = T.Fragment(
    (8, 4),
    forward_fn=lambda i, k, rep: (i + 8 * k + 32 * rep, 0),
    replicate=4,
)


@T.prim_func
def reducer_sum_v2(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
    with T.Kernel(1, threads=128):
        src = T.alloc_fragment((8,), T.float32)
        T.copy(A, src)
        total = T.alloc_reducer((1,), T.float32, op="sum")
        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[0], src[i])
        result = T.alloc_fragment((1,), T.float32)
        T.finalize_reducer(total, result)
        if T.get_thread_binding() == 0:
            B[0] = result[0]


@T.prim_func
def two_reducers_v2(A: T.Tensor((8,), T.float32), B: T.Tensor((2,), T.float32)):
    with T.Kernel(1, threads=128):
        src = T.alloc_fragment((8,), T.float32)
        T.copy(A, src)
        sum_total = T.alloc_reducer((1,), T.float32, op="sum", seed=5.0)
        max_total = T.alloc_reducer((1,), T.float32, op="max")
        T.reducer_init(sum_total)
        T.reducer_init(max_total)
        for i in T.Parallel(8):
            if i < 4:
                T.reducer_update(sum_total[0], src[i])
            T.reducer_update(max_total[0], src[i])
        sum_result = T.alloc_fragment((1,), T.float32)
        max_result = T.alloc_fragment((1,), T.float32)
        T.finalize_reducer(sum_total, sum_result)
        T.finalize_reducer(max_total, max_result)
        if T.get_thread_binding() == 0:
            B[0] = sum_result[0]
            B[1] = max_result[0]


@T.prim_func
def bitwise_reducers_v2(A: T.Tensor((8,), T.int32), B: T.Tensor((3,), T.int32)):
    with T.Kernel(1, threads=128):
        src = T.alloc_fragment((8,), T.int32)
        T.copy(A, src)
        and_total = T.alloc_reducer((1,), T.int32, op="bitand")
        or_total = T.alloc_reducer((1,), T.int32, op="bitor")
        xor_total = T.alloc_reducer((1,), T.int32, op="bitxor")
        T.reducer_init(and_total)
        T.reducer_init(or_total)
        T.reducer_init(xor_total)
        for i in T.Parallel(8):
            T.reducer_update(and_total[0], src[i])
            T.reducer_update(or_total[0], src[i])
            T.reducer_update(xor_total[0], src[i])
        and_result = T.alloc_fragment((1,), T.int32)
        or_result = T.alloc_fragment((1,), T.int32)
        xor_result = T.alloc_fragment((1,), T.int32)
        T.finalize_reducer(and_total, and_result)
        T.finalize_reducer(or_total, or_result)
        T.finalize_reducer(xor_total, xor_result)
        if T.get_thread_binding() == 0:
            B[0] = and_result[0]
            B[1] = or_result[0]
            B[2] = xor_result[0]


@T.prim_func
def unique_owner_reducer_v2(A: T.Tensor((8,), T.float32), B: T.Tensor((8,), T.float32)):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((8,), T.float32, op="sum", seed=2.0)
        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[i], A[i])
        result = T.alloc_fragment((8,), T.float32)
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def unique_owner_serial_reduction_v2(A: T.Tensor((8, 4), T.float32), B: T.Tensor((8,), T.float32)):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((8,), T.float32, op="sum")
        T.reducer_init(total)
        for i in T.Parallel(8):
            for k in T.serial(4):
                T.reducer_update(total[i], A[i, k])
        result = T.alloc_fragment((8,), T.float32)
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def unique_owner_with_global_side_effect_v2(A: T.Tensor((8,), T.float32), B: T.Tensor((8,), T.float32)):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((8,), T.float32, op="sum")
        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[i], A[i])
            B[i] = A[i]
        result = T.alloc_fragment((8,), T.float32)
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def projected_row_reducer_v2(A: T.Tensor((8, 4), T.float32), B: T.Tensor((8,), T.float32)):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((8,), T.float32, op="sum")
        T.reducer_init(total)
        for i, k in T.Parallel(8, 4, loop_layout=_PROJECTED_ROW_LOOP_LAYOUT):
            T.reducer_update(total[i], A[i, k])
        result = T.alloc_fragment((8,), T.float32)
        T.finalize_reducer(total, result)
        T.copy(result, B)


@T.prim_func
def split_projection_groups_v2(
    A: T.Tensor((8,), T.float32),
    C: T.Tensor((16,), T.float32),
    B: T.Tensor((1,), T.float32),
):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((1,), T.float32, op="sum", seed=3.0)
        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[0], A[i])
        for j in T.Parallel(16):
            T.reducer_update(total[0], C[j])
        result = T.alloc_fragment((1,), T.float32)
        T.finalize_reducer(total, result)
        if T.get_thread_binding() == 0:
            B[0] = result[0]


@T.prim_func
def mixed_projection_and_fallback_v2(
    A: T.Tensor((8,), T.float32),
    C: T.Tensor((6,), T.float32),
    B: T.Tensor((1,), T.float32),
):
    with T.Kernel(1, threads=128):
        total = T.alloc_reducer((1,), T.float32, op="sum")
        T.reducer_init(total)
        for i in T.Parallel(8):
            T.reducer_update(total[0], A[i])
        # The first projected implementation only accepts power-of-two
        # collective widths. This site shares one canonical fallback group
        # without degrading the independent 8-way group above.
        for j in T.Parallel(6):
            T.reducer_update(total[0], C[j])
        result = T.alloc_fragment((1,), T.float32)
        T.finalize_reducer(total, result)
        if T.get_thread_binding() == 0:
            B[0] = result[0]


def test_reducer_v2_frontend_ir():
    source = reducer_sum_v2.script()
    assert 'scope="local.reducer"' in source
    assert "T.reducer_init" in source
    assert "T.reducer_update" in source
    assert "T.finalize_reducer" in source


def test_reducer_v2_update_requires_indexed_target():
    with pytest.raises(TypeError, match="indexed local.reducer element"):

        @T.prim_func
        def invalid(A: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=32):
                total = T.alloc_reducer((1,), T.float32)
                T.reducer_init(total)
                # A bare reducer handle is not an indexed update target.
                T.reducer_update(total, A[0])


def test_reducer_v2_update_rejects_non_reducer_target():
    with pytest.raises(ValueError, match="local.reducer target"):

        @T.prim_func
        def invalid(A: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=32):
                total = T.alloc_fragment((1,), T.float32)
                T.reducer_update(total[0], A[0])


def test_reducer_v2_cuda_codegen():
    source = tilelang.compile(
        reducer_sum_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "tl::AllReduce<tl::SumOp, 8, 1" in source
    assert "tl::AllReduce<tl::SumOp, 128" not in source
    assert "local.reducer" not in source


def test_reducer_v2_unique_owner_codegen_uses_local_complete_plan():
    source = tilelang.compile(
        unique_owner_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "tl::AllReduce<" not in source
    assert "NamedBarrier<" not in source
    assert "workspace" not in source
    assert "float total_partial_0[1];" in source


def test_reducer_v2_unique_owner_serial_reduction_codegen():
    source = tilelang.compile(
        unique_owner_serial_reduction_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "tl::AllReduce<" not in source
    assert "NamedBarrier<" not in source
    assert "workspace" not in source
    assert "float total_partial_0[1];" in source


def test_reducer_v2_unique_owner_side_effect_falls_back():
    source = tilelang.compile(
        unique_owner_with_global_side_effect_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "tl::AllReduce<tl::SumOp, 128" in source


def test_reducer_v2_parallel_mk_uses_projected_row_group():
    source = tilelang.compile(
        projected_row_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    # k is carried by thread bits 3-4, so four logical lanes are represented
    # by a 32-thread span with stride 8.
    assert "tl::AllReduce<tl::SumOp, 32, 8" in source
    assert "tl::AllReduce<tl::SumOp, 128" not in source
    assert "float total_partial_0[1];" in source


def test_reducer_v2_incompatible_projections_use_separate_groups():
    source = tilelang.compile(
        split_projection_groups_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "float total_partial_0[1];" in source
    assert "float total_partial_1[1];" in source
    assert "tl::AllReduce<tl::SumOp, 8, 1" in source
    assert "tl::AllReduce<tl::SumOp, 16, 1" in source
    assert "tl::AllReduce<tl::SumOp, 128" not in source


def test_reducer_v2_projected_and_canonical_groups_can_mix():
    source = tilelang.compile(
        mixed_projection_and_fallback_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    ).get_kernel_source()
    assert "float total_partial_0[1];" in source
    assert "float total_partial_1[1];" in source
    assert "tl::AllReduce<tl::SumOp, 8, 1" in source
    assert "tl::AllReduce<tl::SumOp, 128" in source


@tilelang.testing.requires_cuda
def test_reducer_v2_unique_owner_local_complete_correctness():
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    B = tilelang.compile(
        unique_owner_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, A + 2.0, atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_unique_owner_serial_reduction_correctness():
    A = torch.arange(1, 33, dtype=torch.float32, device="cuda").reshape(8, 4)
    B = tilelang.compile(
        unique_owner_serial_reduction_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, A.sum(dim=1), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_projected_row_group_correctness():
    A = torch.arange(1, 33, dtype=torch.float32, device="cuda").reshape(8, 4)
    B = tilelang.compile(
        projected_row_reducer_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, A.sum(dim=1), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_split_projection_groups_correctness():
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    C = torch.arange(1, 17, dtype=torch.float32, device="cuda")
    B = tilelang.compile(
        split_projection_groups_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A, C)
    torch.testing.assert_close(B, (A.sum() + C.sum() + 3.0).reshape(1), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_mixed_projection_and_fallback_correctness():
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    C = torch.arange(1, 7, dtype=torch.float32, device="cuda")
    B = tilelang.compile(
        mixed_projection_and_fallback_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A, C)
    torch.testing.assert_close(B, (A.sum() + C.sum()).reshape(1), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_multiple_reducers_seed_and_predicate():
    A = torch.arange(1, 9, dtype=torch.float32, device="cuda")
    B = tilelang.compile(
        two_reducers_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, torch.tensor([15.0, 8.0], device="cuda"), atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_reducer_v2_builtin_bitwise_reductions():
    A = torch.arange(1, 9, dtype=torch.int32, device="cuda")
    B = tilelang.compile(
        bitwise_reducers_v2,
        out_idx=-1,
        pass_configs=_COMPILE_FLAGS,
    )(A)
    torch.testing.assert_close(B, torch.tensor([0, 15, 8], dtype=torch.int32, device="cuda"))


def test_reducer_v2_rejects_legacy_replication_keyword():
    with pytest.raises(Exception, match="replication"):

        @T.prim_func
        def legacy(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=32):
                total = T.alloc_reducer((1,), T.float32, replication="all")
                T.clear(total)
                B[0] = A[0]


def test_reducer_v2_requires_out_of_place_destination():
    with pytest.raises(Exception, match="destination"):

        @T.prim_func
        def legacy(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=32):
                total = T.alloc_reducer((1,), T.float32)
                T.reducer_init(total)
                T.reducer_update(total[0], A[0])
                T.finalize_reducer(total)
                B[0] = A[0]


def test_reducer_v2_rejects_direct_clear():
    @T.prim_func
    def invalid(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            total = T.alloc_reducer((1,), T.float32)
            T.clear(total)
            T.reducer_init(total)
            T.reducer_update(total[0], A[0])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="forbids ordinary BufferLoad access"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_missing_init():
    @T.prim_func
    def invalid(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_update(total[0], A[0])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="must be dominated by T.reducer_init"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_missing_finalize():
    @T.prim_func
    def invalid(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_init(total)
            T.reducer_update(total[0], A[0])
            B[0] = A[0]

    with pytest.raises(Exception, match="must have exactly one explicit T.reducer_init"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_double_init():
    @T.prim_func
    def invalid(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_init(total)
            T.reducer_init(total)
            T.reducer_update(total[0], A[0])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="must occur exactly once per reducer allocation"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_unprovable_output_index():
    @T.prim_func
    def invalid(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_init(total)
            T.reducer_update(total[1], A[0])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="is not provably within"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_physical_replica_dependent_update():
    @T.prim_func
    def invalid(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((8,), T.float32)
            T.copy(A, src)
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_init(total)
            for i in T.Parallel(8):
                T.reducer_update(total[0], src[i] + T.get_thread_binding())
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="must be invariant across physical replicas"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)


def test_reducer_v2_rejects_private_local_replica_dependent_update():
    @T.prim_func
    def invalid(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=128):
            src = T.alloc_fragment((8,), T.float32)
            T.copy(A, src)
            thread_value = T.alloc_local((1,), T.float32)
            thread_value[0] = A[T.get_thread_binding() % 8]
            total = T.alloc_reducer((1,), T.float32)
            T.reducer_init(total)
            for i in T.Parallel(8):
                T.reducer_update(total[0], src[i] + thread_value[0])
            result = T.alloc_fragment((1,), T.float32)
            T.finalize_reducer(total, result)
            B[0] = result[0]

    with pytest.raises(Exception, match="must be invariant across physical replicas"):
        tilelang.compile(invalid, out_idx=-1, pass_configs=_COMPILE_FLAGS)
