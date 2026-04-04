import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _assert_persistent_gemm_close(kernel, M: int, N: int, K: int) -> None:
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    torch.cuda.synchronize()
    ref = (a.float() @ b.float()).half()
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


def _make_mapping_kernel(
    grid_x: int,
    grid_y: int,
    *,
    order: str | None = None,
    panel_size: int | None = None,
    legacy_order: str | None = None,
    legacy_panel_size: int | None = None,
):
    @T.prim_func
    def func(O: T.Tensor((grid_y, grid_x), T.int32)):
        with T.PersistentKernel(
            grid_x,
            grid_y,
            threads=1,
            order=order,
            panel_size=panel_size,
        ) as (bx, by):
            if legacy_order is not None:
                T.use_swizzle(panel_size=legacy_panel_size, order=legacy_order)
            O[by, bx] = T.int32(1)

    return func


def _lower_with_tile_schedule(func, sm_num: int = 4):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(tvm.target.Target("cuda -arch=sm_80"))(mod)
    mod = tilelang.transform.TileSchedule(sm_num)(mod)
    return mod["main"]


def _extract_store_indices(func):
    state = {"w_var": None, "block_var": None, "indices": None, "has_legacy_attr": False}

    def visitor(node):
        if isinstance(node, tvm.tir.For) and node.loop_var.name == "w_tile_sched":
            state["w_var"] = node.loop_var
        elif isinstance(node, tvm.tir.AttrStmt) and node.attr_key == "thread_extent":
            iter_var = node.node
            if isinstance(iter_var, tvm.tir.IterVar) and iter_var.thread_tag == "blockIdx.x":
                state["block_var"] = iter_var.var
        elif isinstance(node, tvm.tir.AttrStmt) and node.attr_key == "threadblock_swizzle_pattern":
            state["has_legacy_attr"] = True
        elif isinstance(node, tvm.tir.BufferStore) and node.buffer.name == "O":
            state["indices"] = node.indices

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    assert state["w_var"] is not None
    assert state["block_var"] is not None
    assert state["indices"] is not None
    return state


def _eval_index(expr, subst):
    return int(tvm.arith.Analyzer().simplify(tvm.tir.stmt_functor.substitute(expr, subst)))


def _expected_row(tile_linear: int, grid_x: int, grid_y: int, panel_size: int):
    grid_size = grid_x * grid_y
    panel_span = panel_size * grid_x
    panel_offset = tile_linear % panel_span
    panel_idx = tile_linear // panel_span
    total_panel = (grid_size + panel_span - 1) // panel_span
    stride = panel_size if panel_idx + 1 < total_panel else (grid_size - panel_idx * panel_span) // grid_x
    col_idx = grid_x - 1 - panel_offset // stride if panel_idx & 1 else panel_offset // stride
    row_idx = panel_offset % stride + panel_idx * panel_size
    return row_idx, col_idx


def _expected_column(tile_linear: int, grid_x: int, grid_y: int, panel_size: int):
    grid_size = grid_x * grid_y
    panel_span = panel_size * grid_y
    panel_offset = tile_linear % panel_span
    panel_idx = tile_linear // panel_span
    total_panel = (grid_size + panel_span - 1) // panel_span
    stride = panel_size if panel_idx + 1 < total_panel else (grid_size - panel_idx * panel_span) // grid_y
    row_idx = grid_y - 1 - panel_offset // stride if panel_idx & 1 else panel_offset // stride
    col_idx = panel_offset % stride + panel_idx * panel_size
    return row_idx, col_idx


def _compile_persistent_gemm(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    *,
    target: str,
    num_stages=3,
    order: str | None = None,
    panel_size: int | None = None,
    dtype=T.float16,
    accum_dtype=T.float32,
):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.PersistentKernel(
            T.ceildiv(N, block_N),
            T.ceildiv(M, block_M),
            threads=128,
            order=order,
            panel_size=panel_size,
        ) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return tilelang.compile(gemm, out_idx=[-1], target=target)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("num_stages", [2, 3])
def test_persistent_annotated_gemm_compiles_for_supported_pipeline_stages(num_stages):
    M = N = K = 1024
    block_M = block_N = 128
    block_K = 32
    kernel = _compile_persistent_gemm(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        num_stages=num_stages,
        target="cuda",
    )

    source = kernel.get_kernel_source()
    assert "gemm_kernel" in source
    _assert_persistent_gemm_close(kernel, M, N, K)


@tilelang.testing.requires_cuda
def test_persistent_annotated_gemm_stage3_waits_keep_persistent_phase_progress():
    kernel_codegen = _compile_persistent_gemm(
        8192,
        8192,
        8192,
        128,
        128,
        32,
        num_stages=3,
        target="cuda",
    )

    source = kernel_codegen.get_kernel_source()
    assert "mbarrier[4].wait((w_tile_sched & 1));" in source
    assert "mbarrier[5].wait((w_tile_sched & 1));" in source
    assert "mbarrier[0].wait(1);" in source
    assert "mbarrier[1].wait(1);" in source

    M = N = K = 1024
    kernel_check = _compile_persistent_gemm(
        M,
        N,
        K,
        128,
        128,
        32,
        num_stages=3,
        target="cuda",
    )
    _assert_persistent_gemm_close(kernel_check, M, N, K)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(8, 0)
@pytest.mark.parametrize("order", ["row", "column"])
def test_persistent_annotated_gemm_supports_swizzled_tile_order(order):
    M = N = K = 1024
    kernel = _compile_persistent_gemm(
        M,
        N,
        K,
        128,
        128,
        32,
        num_stages=2,
        order=order,
        panel_size=8,
        target="cuda",
    )
    _assert_persistent_gemm_close(kernel, M, N, K)


def test_tile_schedule_swizzles_row_major_persistent_kernel():
    grid_x, grid_y, sm_num = 4, 3, 4
    func = _make_mapping_kernel(grid_x, grid_y, order="row", panel_size=2)
    lowered = _lower_with_tile_schedule(func, sm_num=sm_num)
    info = _extract_store_indices(lowered)

    for tile_linear in range(grid_x * grid_y):
        w = tile_linear // sm_num
        block = tile_linear % sm_num
        subst = {
            info["w_var"]: tvm.tir.IntImm("int32", w),
            info["block_var"]: tvm.tir.IntImm("int32", block),
        }
        actual = tuple(_eval_index(idx, subst) for idx in info["indices"])
        assert actual == _expected_row(tile_linear, grid_x, grid_y, 2)


def test_tile_schedule_swizzles_column_major_persistent_kernel():
    grid_x, grid_y, sm_num = 4, 3, 4
    func = _make_mapping_kernel(grid_x, grid_y, order="column", panel_size=2)
    lowered = _lower_with_tile_schedule(func, sm_num=sm_num)
    info = _extract_store_indices(lowered)

    for tile_linear in range(grid_x * grid_y):
        w = tile_linear // sm_num
        block = tile_linear % sm_num
        subst = {
            info["w_var"]: tvm.tir.IntImm("int32", w),
            info["block_var"]: tvm.tir.IntImm("int32", block),
        }
        actual = tuple(_eval_index(idx, subst) for idx in info["indices"])
        assert actual == _expected_column(tile_linear, grid_x, grid_y, 2)


def test_persistent_kernel_swizzle_overrides_legacy_use_swizzle_annotation():
    func = _make_mapping_kernel(
        4,
        3,
        order="row",
        panel_size=2,
        legacy_order="column",
        legacy_panel_size=3,
    )
    lowered = _lower_with_tile_schedule(func, sm_num=4)
    info = _extract_store_indices(lowered)
    assert info["has_legacy_attr"] is False

    subst = {
        info["w_var"]: tvm.tir.IntImm("int32", 1),
        info["block_var"]: tvm.tir.IntImm("int32", 0),
    }
    actual = tuple(_eval_index(idx, subst) for idx in info["indices"])
    assert actual == _expected_row(4, 4, 3, 2)
    assert actual != _expected_column(4, 4, 3, 3)


def test_tile_schedule_uses_legacy_use_swizzle_when_persistent_kernel_has_no_override():
    func = _make_mapping_kernel(
        4,
        3,
        legacy_order="column",
        legacy_panel_size=3,
    )
    lowered = _lower_with_tile_schedule(func, sm_num=4)
    info = _extract_store_indices(lowered)
    assert info["has_legacy_attr"] is False

    subst = {
        info["w_var"]: tvm.tir.IntImm("int32", 1),
        info["block_var"]: tvm.tir.IntImm("int32", 0),
    }
    actual = tuple(_eval_index(idx, subst) for idx in info["indices"])
    assert actual == _expected_column(4, 4, 3, 3)


if __name__ == "__main__":
    tilelang.testing.main()
