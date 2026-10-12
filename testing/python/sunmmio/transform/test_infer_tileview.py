"""Tests for automatic TileView inference in LowerTilesLoop.

Verifies that when no T.annotate_tileview() is used, LowerTilesLoop
derives feasible TileView candidates from each buffer access and picks a
common execution plan with the expected tile_size / execution_domain_axes.
"""

import pytest
import tilelang
import tilelang as tl
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.layout import CuteLayout, make_aligned_row_major, make_zz_layout
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm import tir
from tvm import IRModule


def apply_sunmmio_passes(mod, target):
    """Apply the full SUNMMIO pass pipeline used for DMA copy lowering."""
    mod = tvm.tir.transform.BindTarget(target)(mod)
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    mod = tilelang.transform.InjectAssumes()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.InferSramScope()(mod)
    mod = tilelang.transform.LegalizeSunmmioDataPath()(mod)
    mod = tilelang.transform.LayoutReducer()(mod)
    mod = tilelang.transform.SunmmioLayoutInference()(mod)
    mod = tilelang.transform.LowerTileOp()(mod)
    mod = tl.transform.LowerTilesLoop()(mod)
    return mod


# ---------------------------------------------------------
# Helper: collect tile loop annotations from IR
# ---------------------------------------------------------
def collect_tile_annotations(func):
    """Return the scope-root plan annotations and execution-axis count."""
    scope_root = None
    execution_axes = []

    def visit(stmt):
        if isinstance(stmt, tir.For):
            ann = stmt.annotations
            if ann is None:
                return
            nonlocal scope_root
            if "tile.domain" in ann:
                scope_root = {
                    "tile.domain": ann["tile.domain"],
                    "tile.tile_size": ann["tile.tile_size"],
                    "tile.execution_domain_axes": ann["tile.execution_domain_axes"],
                }
            if "tile.execution_axis" in ann:
                execution_axes.append(int(ann["tile.execution_axis"]))

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    execution_axes.sort()
    return scope_root, execution_axes


def _to_int_list(arr):
    """Convert an Array<PrimExpr> annotation to a Python list of ints."""
    return [int(x) for x in arr]


def assert_scope_plan(mod, expected_tile_size, expected_execution_domain_axes):
    scope_root, execution_axes = collect_tile_annotations(mod["main"])

    assert scope_root is not None
    assert execution_axes == list(range(len(expected_tile_size)))
    assert _to_int_list(scope_root["tile.tile_size"]) == list(expected_tile_size)
    assert _to_int_list(scope_root["tile.execution_domain_axes"]) == list(expected_execution_domain_axes)


def collect_scope_plans(func):
    plans = []

    def visit(stmt):
        if not isinstance(stmt, tir.For) or stmt.annotations is None or "tile.domain" not in stmt.annotations:
            return
        plans.append(
            (
                _to_int_list(stmt.annotations["tile.tile_size"]),
                _to_int_list(stmt.annotations["tile.execution_domain_axes"]),
            )
        )

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return plans


def collect_loads(func, buffer_name):
    loads = []

    def visit(stmt, loads=loads):
        if isinstance(stmt, tir.BufferLoad) and stmt.buffer.name == buffer_name:
            loads.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return loads


def collect_stores(func, buffer_name):
    stores = []

    def visit(stmt, stores=stores):
        if isinstance(stmt, tir.BufferStore) and stmt.buffer.name == buffer_name:
            stores.append(stmt)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return stores


def collect_if_conditions(func):
    conditions = []

    def visit(stmt, conditions=conditions):
        if isinstance(stmt, tir.IfThenElse):
            conditions.append(stmt.condition)

    tvm.tir.stmt_functor.post_order_visit(func.body, visit)
    return conditions


def assert_preserved_mixed_rank_load(mod, expected_tile_size):
    script = mod["main"].script()
    assert "T.sunmmio_unaligned_tile_load" not in script

    loads = collect_loads(mod["main"], "B_shared")
    assert loads, "Expected mixed-rank side access to remain a BufferLoad"
    assert all(load.predicate is None for load in loads)

    tile_height = expected_tile_size[0]
    assert any(f"* {tile_height}" in str(load.indices[0]) or f"*{tile_height}" in str(load.indices[0]) for load in loads)


# ---------------------------------------------------------
# Test 1: 2D T.Tiles without annotate_tileview
# ---------------------------------------------------------
def test_infer_tileview_2d_no_annotation():
    """Row-major 2D pointwise access should choose the largest full-row tile."""
    M, N = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            C_shared = T.alloc_shared((M, N), "float16")

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            # No annotate_tileview — should be auto-inferred
            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[2, 128], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_2d_with_layout_annotation():
    """Blockwise 2D pointwise access should use one full ZZ block."""
    M, N = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            C_shared = T.alloc_shared((M, N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                    B_shared: make_zz_layout(B_shared),
                    C_shared: make_zz_layout(C_shared),
                }
            )

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            # No annotate_tileview — should be auto-inferred
            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[32, 32], expected_execution_domain_axes=[0, 1])


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_infer_tileview_layout_bounded_zz_uses_full_inner_block(dtype):
    """Execution TileView uses one full ZZ block regardless of dtype."""
    M, N = 128, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M, N), dtype)

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                    B_shared: make_zz_layout(B_shared),
                }
            )

            T.copy(A, A_shared)
            for i, j in T.Tiles([M, N], parallel=True):
                B_shared[i, j] = A_shared[i, j] + A_shared[i, j]
            T.copy(B_shared, B)

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[32, 32], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_layout_bounded_rank1_uses_covered_extent_and_predicate():
    """A padded row-major buffer tiles its covered extent and masks logical padding."""
    N = 1000

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float16"),
        B: T.Tensor((N,), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((N,), "float16")
            B_shared = T.alloc_shared((N,), "float16")

            T.annotate_layout(
                {
                    A_shared: make_aligned_row_major((N,), "float16", align_bytes=64),
                    B_shared: make_aligned_row_major((N,), "float16", align_bytes=64),
                }
            )

            T.copy(A, A_shared)
            for i in T.Tiles([N], parallel=True):
                B_shared[i] = A_shared[i] + A_shared[i]
            T.copy(B_shared, B)

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=[1024], expected_execution_domain_axes=[0])
    stores = collect_stores(mod["main"], "B_shared")
    assert stores, "Expected lowered B_shared stores"
    assert all(store.predicate is not None for store in stores)
    assert all("< 1000" in str(store.predicate) or "<1000" in str(store.predicate) for store in stores)
    assert not collect_if_conditions(mod["main"]), "Every lane scope is a partial tile"


def test_infer_tileview_layout_bounded_rowmajor_uses_full_2d_region():
    """A dense aligned row-major execution region is one semantic tile."""
    M, N = 8, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_aligned_row_major((M, N), "float16", align_bytes=64),
                    B_shared: make_aligned_row_major((M, N), "float16", align_bytes=64),
                }
            )

            T.copy(A, A_shared)
            for i, j in T.Tiles([M, N], parallel=True):
                B_shared[i, j] = A_shared[i, j] + A_shared[i, j]
            T.copy(B_shared, B)

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[8, 128], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_layout_bounded_rowmajor_uses_padded_2d_region():
    """A 2D row-major TileView spans covered padding and masks logical width."""
    M, N = 8, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_aligned_row_major((M, N), "float16", align_bytes=1024),
                    B_shared: make_aligned_row_major((M, N), "float16", align_bytes=1024),
                }
            )

            T.copy(A, A_shared)
            for i, j in T.Tiles([M, N], parallel=True):
                B_shared[i, j] = A_shared[i, j] + A_shared[i, j]
            T.copy(B_shared, B)

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=[8, 512], expected_execution_domain_axes=[0, 1])
    stores = collect_stores(mod["main"], "B_shared")
    assert stores, "Expected lowered B_shared stores"
    assert all(store.predicate is not None for store in stores)
    assert all("< 128" in str(store.predicate) or "<128" in str(store.predicate) for store in stores)
    assert all("< 8" not in str(store.predicate) and "<8" not in str(store.predicate) for store in stores)
    assert not collect_if_conditions(mod["main"]), "The only execution tile is width-partial"


def test_infer_tileview_non_rowmajor_single_level_layout_stays_register_bounded():
    """A single-level column-major layout must not enter row-major policy."""
    M, N = 8, 128
    column_major = CuteLayout(
        [M, N],
        [M, N],
        [1, M],
        [1, 1],
    )._inner

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            T.annotate_layout({A_shared: column_major, B_shared: column_major})

            for i, j in T.Tiles([M, N], parallel=True):
                B_shared[i, j] = A_shared[i, j] + A_shared[i, j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[8, 32], expected_execution_domain_axes=[0, 1])


# ---------------------------------------------------------
# Test 2: 1D T.Tiles without annotate_tileview
# ---------------------------------------------------------
def test_infer_tileview_1d_no_annotation():
    """Layout-inferred 1D row-major buffers use the full covered extent."""
    N = 1024

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((N,), "float32")
            B_shared = T.alloc_shared((N,), "float32")

            T.copy(A[0:N], A_shared)

            for i in T.Tiles([N], parallel=True):
                B_shared[i] = A_shared[i] * A_shared[i]

            T.copy(B_shared, B[0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[1024], expected_execution_domain_axes=[0])


def test_infer_rank1_tileview_from_2d_buffer_access():
    """Layout inference may collapse a trailing-dim access to a blockwise rank-1 tile."""
    M, N = 64, 256

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((N,), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float32")
            B_shared = T.alloc_shared((N,), "float32")

            T.copy(A[0:M, 0:N], A_shared)

            for i in T.Tiles([N], parallel=True):
                B_shared[i] = A_shared[0, i] * A_shared[0, i]

            T.copy(B_shared, B[0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[32], expected_execution_domain_axes=[0])


def test_infer_rank1_tileview_from_2d_buffer_access_with_outer_loop_var():
    """An outer serial loop var stays outside the 1D tile domain after layout inference."""
    M, N = 32, 256

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float32")
            B_shared = T.alloc_shared((M, N), "float32")

            T.copy(A[0:M, 0:N], A_shared)

            for row in T.serial(M):
                for j in T.Tiles([N], parallel=True):
                    B_shared[row, j] = A_shared[row, j] * A_shared[row, j]

            T.copy(B_shared, B[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[32], expected_execution_domain_axes=[0])


@pytest.mark.parametrize("dtype", ["bfloat16", "float32"])
def test_infer_complete_rank1_nondivisor_domain(dtype):
    """Original rank-1 scopes may keep a complete non-divisor domain tile."""
    storage_width = 64

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((storage_width,), dtype, scope="shared.rsram")
            dst = T.alloc_shared((storage_width,), dtype, scope="shared.rsram")
            layout = make_aligned_row_major((storage_width,), dtype, align_bytes=64)
            T.annotate_layout({src: layout, dst: layout})

            for segment in T.serial(4):
                for j in T.Tiles([14], parallel=True):
                    dst[segment * 14 + j] = src[segment * 14 + j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=[14], expected_execution_domain_axes=[0])


def test_fp16_nondivisor_rank1_keeps_legacy_plan():
    """FP16 remains outside the cross-carrier aligned-1D bridge."""
    storage_width = 64

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((storage_width,), "float16", scope="shared.rsram")
            dst = T.alloc_shared((storage_width,), "float16", scope="shared.rsram")
            layout = make_aligned_row_major((storage_width,), "float16", align_bytes=64)
            T.annotate_layout({src: layout, dst: layout})

            for segment in T.serial(4):
                for j in T.Tiles([14], parallel=True):
                    dst[segment * 14 + j] = src[segment * 14 + j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=[2], expected_execution_domain_axes=[0])


def test_zz_nondivisor_rank1_keeps_legacy_plan():
    """Hierarchical layouts do not receive the flat-layout relaxation."""

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((32, 64), "bfloat16", scope="shared.rsram")
            dst = T.alloc_shared((32, 64), "bfloat16", scope="shared.rsram")
            layout = make_zz_layout((32, 64), [0, 1], (32, 32))
            T.annotate_layout({src: layout, dst: layout})

            for segment in T.serial(4):
                for j in T.Tiles([14], parallel=True):
                    dst[0, segment * 14 + j] = src[0, segment * 14 + j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=[2], expected_execution_domain_axes=[0])


def test_rank2_domain_does_not_enable_nondivisor_side_bridge():
    """Rank reduction from an original 2D scope keeps the legacy search space."""

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            matrix = T.alloc_shared((32, 32), "bfloat16", scope="shared.rsram")
            side = T.alloc_shared((28,), "bfloat16", scope="shared.rsram")
            T.annotate_layout(
                {
                    matrix: make_zz_layout((32, 32), [0, 1], (32, 32)),
                    side: make_aligned_row_major((28,), "bfloat16", align_bytes=64),
                }
            )

            for i, j in T.Tiles([4, 14], parallel=True):
                matrix[i, j] = matrix[i, j] + side[j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert all(tile_size != [4, 14] for tile_size, _ in collect_scope_plans(mod["main"]))


# ---------------------------------------------------------
# Test 3: Mixed-rank (1D + 2D) in same T.Tiles
# ---------------------------------------------------------
@pytest.mark.parametrize(
    "dtype,expected_tile_size",
    [
        ("float16", [32, 32]),
        ("bfloat16", [4, 4]),
        ("float32", [4, 4]),
    ],
)
def test_exact_small_2d_fallback_uses_domain_sized_carrier_plan(dtype, expected_tile_size):
    """The motivating 4x4 domain avoids oversubscribing to the full ZZ block."""
    matrix_shape = (64, 64)
    vector_shape = (500,)

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared(matrix_shape, dtype, scope="shared.rsram")
            B_shared = T.alloc_shared(vector_shape, dtype, scope="shared.rsram")
            T.annotate_layout(
                {
                    A_shared: make_zz_layout(matrix_shape, [0, 1], (32, 32)),
                    B_shared: make_aligned_row_major(vector_shape, dtype, align_bytes=64),
                }
            )

            for i, j in T.Tiles([4, 4], parallel=True):
                A_shared[i, j] = A_shared[i, j] * B_shared[i]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=expected_tile_size, expected_execution_domain_axes=[0, 1])


@pytest.mark.parametrize(
    "dtype,matrix_shape,expected_tile_size",
    [
        ("float32", (64, 32), [4, 4]),
        ("float32", (64, 64), [64, 64]),
        ("float16", (64, 64), [64, 64]),
        ("bfloat16", (64, 64), [4, 4]),
        ("bfloat16", (64, 128), [64, 128]),
    ],
)
def test_row_major_small_2d_fallback_requires_one_complete_carrier(dtype, matrix_shape, expected_tile_size):
    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared(matrix_shape, dtype, scope="shared.rsram")
            dst = T.alloc_shared(matrix_shape, dtype, scope="shared.rsram")
            layout = make_aligned_row_major(matrix_shape, dtype, align_bytes=64)
            T.annotate_layout({src: layout, dst: layout})

            for i, j in T.Tiles([4, 4], parallel=True):
                dst[i, j] = src[i, j] * 2.0

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=expected_tile_size, expected_execution_domain_axes=[0, 1])


def test_small_2d_fallback_does_not_override_manual_tileview():
    from tilelang.tileview import make_tileview

    shape = (64, 64)

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared(shape, "float32", scope="shared.rsram")
            dst = T.alloc_shared(shape, "float32", scope="shared.rsram")
            layout = make_zz_layout(shape, [0, 1], (32, 32))
            T.annotate_layout({src: layout, dst: layout})
            T.annotate_tileview({src: make_tileview(src, (32, 32), (-2, -1))})

            for i, j in T.Tiles([4, 4], parallel=True):
                dst[i, j] = src[i, j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=[32, 32], expected_execution_domain_axes=[0, 1])


def test_dynamic_domain_does_not_select_static_small_2d_plan():
    shape = (64, 64)

    @T.prim_func
    def main(n: T.int32):
        with T.Kernel(1, threads=128):
            src = T.alloc_shared(shape, "float32", scope="shared.rsram")
            dst = T.alloc_shared(shape, "float32", scope="shared.rsram")
            layout = make_zz_layout(shape, [0, 1], (32, 32))
            T.annotate_layout({src: layout, dst: layout})

            for i, j in T.Tiles([n, 4], parallel=True):
                dst[i, j] = src[i, j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=[32, 32], expected_execution_domain_axes=[0, 1])


@pytest.mark.parametrize(
    ("dtype", "expected_tile_size"),
    [
        ("float32", [32, 32]),
        ("float16", [32, 32]),
        ("bfloat16", [32, 32]),
    ],
)
def test_infer_tileview_mixed_rank_load(dtype, expected_tile_size):
    """Mixed-rank rank-1 loads stay as logical BufferLoad in LowerTilesLoop.

    B_shared is 1D and tiled along the height axis. Strict TileView search
    rejects tile width 1 because 64-byte RSRAM alignment requires multiple
    elements, then the fallback search allows the side load. The side load does
    not cap the execution TileView; its register-bounded carrier repair remains
    deferred to Sunmmio codegen.
    """
    M, N = 128, 64

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)
            C_shared = T.alloc_shared((M, N), dtype)

            T.copy(A[0:M, 0:N], A_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] + B_shared[i]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=expected_tile_size, expected_execution_domain_axes=[0, 1])
    assert_preserved_mixed_rank_load(mod, expected_tile_size)


@pytest.mark.parametrize(
    ("dtype", "expected_tile_size"),
    [
        ("float32", [32, 32]),
        ("float16", [32, 32]),
        ("bfloat16", [32, 32]),
    ],
)
def test_infer_tileview_mixed_rank_load_inside_exp2(dtype, expected_tile_size):
    """Preserved mixed-rank rank-1 BufferLoad can feed another TIR PrimExpr op."""
    M, N = 128, 64

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)
            C_shared = T.alloc_shared((M, N), dtype)

            T.copy(A[0:M, 0:N], A_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] + T.exp2(B_shared[i])

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_scope_plan(mod, expected_tile_size=expected_tile_size, expected_execution_domain_axes=[0, 1])
    script = mod["main"].script()
    assert "T.exp2" in script
    assert_preserved_mixed_rank_load(mod, expected_tile_size)


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
def test_infer_tileview_mixed_rank_store_still_rejected(dtype):
    """The alignment-relaxed fallback is load-only, not store repair."""
    M, N = 128, 64

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), dtype)
            B_shared = T.alloc_shared((M,), dtype)
            C_shared = T.alloc_shared((M, N), dtype)

            T.copy(A[0:M, 0:N], A_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                B_shared[i] = A_shared[i, j]
                C_shared[i, j] = A_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with (
        tvm.target.Target(target),
        pytest.raises(
            tvm.error.InternalError,
            match="Implicit reduction in T.Tiles is not supported.",
        ),
    ):
        apply_sunmmio_passes(mod, target)


def test_infer_tileview_falls_back_to_1d_for_scalar_side_load():
    """A row-dependent scalar side load must not reject a 2D T.Tiles scope."""
    M, N = 32, 64

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            x_shared = T.alloc_shared((M, N), "float32")
            post_mix_shared = T.alloc_shared((M, M), "float32")
            output_accum = T.alloc_shared((M, N), "float32")

            for bx, by in T.Tiles([M, N], parallel=True):
                output_accum[bx, by] = x_shared[bx, by] * post_mix_shared[bx, 0]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)

    assert_scope_plan(mod, expected_tile_size=[64], expected_execution_domain_axes=[1])
    loads = collect_loads(mod["main"], "post_mix_shared")
    assert loads
    assert all(len(load.indices) == 2 for load in loads)


def test_infer_tileview_keeps_2d_for_trailing_axis_side_load():
    """A side load along the contiguous trailing axis supports the 2D plan."""
    M, N = 32, 64

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            x_shared = T.alloc_shared((M, N), "float32")
            post_mix_shared = T.alloc_shared((M, M), "float32")
            output_accum = T.alloc_shared((M, N), "float32")

            for bx, by in T.Tiles([M, N], parallel=True):
                output_accum[bx, by] = x_shared[bx, by] * post_mix_shared[0, by]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)

    assert_scope_plan(mod, expected_tile_size=[1, 32], expected_execution_domain_axes=[0, 1])
    loads = collect_loads(mod["main"], "post_mix_shared")
    assert loads
    assert all(len(load.indices) == 2 for load in loads)


def test_infer_tileview_falls_back_to_rank1_for_packed_2d_to_1d_store():
    """A fused packed store keeps its unit-stride axis as a 1D tile."""
    H, BASE, VECTOR_SIZE, MATRIX_SIZE = 4, 8, 128, 32

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((VECTOR_SIZE,), "float32")
            B_shared = T.alloc_shared((MATRIX_SIZE, MATRIX_SIZE), "float32")

            for i, j in T.Tiles([H, H], parallel=True):
                A_shared[BASE + i * H + j] = B_shared[i, j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)

    assert_scope_plan(mod, expected_tile_size=[H], expected_execution_domain_axes=[1])


def test_infer_tileview_falls_back_to_rank1_for_packed_1d_to_2d_load():
    """A fused packed load keeps its unit-stride axis as a 1D tile."""
    H, BASE, VECTOR_SIZE, MATRIX_SIZE = 4, 8, 128, 32

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((VECTOR_SIZE,), "float32")
            B_shared = T.alloc_shared((MATRIX_SIZE, MATRIX_SIZE), "float32")

            for i, j in T.Tiles([H, H], parallel=True):
                B_shared[i, j] = A_shared[BASE + i * H + j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)

    assert_scope_plan(mod, expected_tile_size=[H], expected_execution_domain_axes=[1])


def test_rank1_fallback_preserves_dynamic_tile_domain():
    """Rank reduction keeps dynamic domain expressions in the tile scope."""
    H, BASE, VECTOR_SIZE, MATRIX_SIZE = 4, 8, 128, 32

    @T.prim_func
    def main(valid: T.int32):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((VECTOR_SIZE,), "float32")
            B_shared = T.alloc_shared((MATRIX_SIZE, MATRIX_SIZE), "float32")

            for i, j in T.Tiles([H, T.min(H, valid)], parallel=True):
                B_shared[i, j] = A_shared[BASE + i * H + j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)

    assert_scope_plan(mod, expected_tile_size=[H], expected_execution_domain_axes=[1])
    scope_root, _ = collect_tile_annotations(mod["main"])
    assert scope_root is not None
    assert "valid" in str(scope_root["tile.domain"][1])


def test_infer_tileview_keeps_scalar_fallback_when_no_axis_is_unit_stride():
    """Rank reduction must not reinterpret a strided access as a dense tile."""
    H, BASE, VECTOR_SIZE, MATRIX_SIZE = 4, 8, 128, 32

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((VECTOR_SIZE,), "float32")
            B_shared = T.alloc_shared((MATRIX_SIZE, MATRIX_SIZE), "float32")

            for i, j in T.Tiles([H, H], parallel=True):
                B_shared[i, j] = A_shared[BASE + i * H + j * 2]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    script = mod["main"].script()

    assert "tile.domain" not in script
    assert "tile.scope_entry" not in script
    assert "for i, j in T.grid(4, 4)" in script


def test_infer_tileview_keeps_scalar_fallback_when_index_mixes_tile_axes():
    """An index that mixes both tile axes cannot be represented by a TileView."""
    domain, shape = 4, 16

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((shape, shape), "float32")
            dst = T.alloc_shared((shape, shape), "float32")

            for i, j in T.Tiles([domain, domain], parallel=True):
                dst[i, j] = src[i + j, i + j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    script = mod["main"].script()

    assert "tile.domain" not in script
    assert "tile.scope_entry" not in script
    assert "for i, j in T.grid(4, 4)" in script


def test_infer_tileview_keeps_scalar_fallback_for_incompatible_axis_order():
    """Transposed source and row-major destination accesses share no execution plan."""
    domain, shape = 4, 16

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((shape, shape), "float32")
            dst = T.alloc_shared((shape, shape), "float32")

            for i, j in T.Tiles([domain, domain], parallel=True):
                dst[i, j] = src[j, i]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    script = mod["main"].script()

    assert "tile.domain" not in script
    assert "tile.scope_entry" not in script
    assert "for i, j in T.grid(4, 4)" in script


@pytest.mark.parametrize(
    "dtype",
    [
        pytest.param(T.float4_e2m1fn, id="fp4"),
        pytest.param(T.mxfp4, id="mxfp4"),
        pytest.param(T.mxfp8, id="mxfp8"),
    ],
)
def test_subbyte_and_mx_dtypes_skip_aligned_1d_bridge(dtype, capfd):
    """Unverified 1D dtype paths must remain on the conservative fallback."""
    width = 32

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            src = T.alloc_shared((256,), dtype, scope="shared.rsram")
            dst = T.alloc_shared((256,), dtype, scope="shared.rsram")

            for j in T.Tiles([width], parallel=True):
                dst[j] = src[width + j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    captured = capfd.readouterr()
    script = mod["main"].script()

    assert "Skipping aligned 1D bridge candidates" in captured.err
    assert "tile.domain" not in script
    assert "tile.scope_entry" not in script
    assert "for j in range(32)" in script


def test_rank1_fallback_does_not_override_rank2_manual_tileview():
    """Rank reduction must preserve an incompatible explicit 2D TileView."""
    from tilelang.tileview import make_tileview

    H, BASE, VECTOR_SIZE, MATRIX_SIZE = 4, 8, 128, 32

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((VECTOR_SIZE,), "float32")
            B_shared = T.alloc_shared((MATRIX_SIZE, MATRIX_SIZE), "float32")
            T.annotate_tileview({B_shared: make_tileview(B_shared, (H, MATRIX_SIZE), (-2, -1))})

            for i, j in T.Tiles([H, H], parallel=True):
                B_shared[i, j] = A_shared[BASE + i * H + j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    script = mod["main"].script()

    assert "tile.domain" not in script
    assert "tile.scope_entry" not in script
    assert "for i, j in T.grid(4, 4)" in script


def test_infer_subaligned_rank1_tile_for_serialized_2d_copy():
    """A serialized outer axis leaves a bridgeable logical 1D tile."""
    H = 4

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            cm = T.alloc_shared((H, H), "float32", scope="shared.rsram")
            comb = T.alloc_shared((1, H * H), "float32", scope="shared.rsram")
            T.annotate_layout(
                {
                    cm: make_zz_layout((H, H), [0, 1], (32, 32)),
                    comb: make_aligned_row_major((1, H * H), "float32", align_bytes=64),
                }
            )

            for j in T.serial(H):
                for k in T.Tiles([H], parallel=True):
                    comb[0, j * H + k] = cm[j, k]

    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = IRModule.from_expr(main.with_attr("global_symbol", "main").with_attr("target", target))
    mod = tl.transform.LowerTilesLoop()(mod)

    assert_scope_plan(mod, expected_tile_size=[H], expected_execution_domain_axes=[0])


def test_infer_scope_local_subaligned_then_direct_rank1_tiles():
    """The same temp buffer can use a bridged tile then a direct full row."""
    H, W = 4, 32

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            cm = T.alloc_shared((H, W), "float32", scope="shared.rsram")
            m_shared = T.alloc_shared((W,), "float32", scope="shared.rsram")
            temp = T.alloc_shared((W,), "float32", scope="shared.rsram")
            T.annotate_layout(
                {
                    cm: make_zz_layout((H, W), [0, 1], (32, 32)),
                    m_shared: make_aligned_row_major((W,), "float32", align_bytes=64),
                    temp: make_aligned_row_major((W,), "float32", align_bytes=64),
                }
            )

            for i in T.serial(H):
                for j in T.Tiles([H], parallel=True):
                    temp[j] = m_shared[i * H + j]
                for j in T.Tiles([W], parallel=True):
                    cm[i, j] = temp[j]

    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = IRModule.from_expr(main.with_attr("global_symbol", "main").with_attr("target", target))
    mod = tl.transform.LowerTilesLoop()(mod)

    assert sorted(collect_scope_plans(mod["main"])) == [([H], [0]), ([W], [0])]


def test_infer_tileview_swapped_domain_binding():
    """Execution axes should be inferred from access bindings, not loop order."""
    M, N = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            C_shared = T.alloc_shared((M, N), "float16")

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            for j, i in T.Tiles([N, M], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[32, 32], expected_execution_domain_axes=[1, 0])


# ---------------------------------------------------------
# Test 4: Manual annotation overrides inference
# ---------------------------------------------------------
def test_manual_annotation_overrides_inference():
    """When T.annotate_tileview is provided, it overrides inference.

    Without annotation, blockwise inference would produce tile_size=(32, 32).
    With annotation specifying a smaller but legal blockwise tile, we should
    preserve that override.
    """
    from tilelang.tileview import make_tileview

    M, N = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            C_shared = T.alloc_shared((M, N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                    C_shared: make_zz_layout(C_shared),
                }
            )

            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, (4, 32), (-2, -1)),
                    C_shared: make_tileview(C_shared, (4, 32), (-2, -1)),
                }
            )

            T.copy(A[0:M, 0:N], A_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * A_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[4, 32], expected_execution_domain_axes=[0, 1])


def test_manual_layout_bounded_zz_accepts_full_inner_block():
    """A manual execution TileView may use one full ZZ block."""
    from tilelang.tileview import make_tileview

    M, N = 128, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                    B_shared: make_zz_layout(B_shared),
                }
            )
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, (32, 32), (-2, -1)),
                    B_shared: make_tileview(B_shared, (32, 32), (-2, -1)),
                }
            )

            T.copy(A, A_shared)
            for i, j in T.Tiles([M, N], parallel=True):
                B_shared[i, j] = A_shared[i, j] + A_shared[i, j]
            T.copy(B_shared, B)

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[32, 32], expected_execution_domain_axes=[0, 1])


def test_manual_layout_bounded_rowmajor_accepts_covered_shape():
    """A manual execution TileView may use the padded row-major envelope."""
    from tilelang.tileview import make_tileview

    M, N = 8, 128
    layout = make_aligned_row_major((M, N), "float16", align_bytes=1024)

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            T.annotate_layout({A_shared: layout, B_shared: layout})
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, (M, 512), (-2, -1)),
                    B_shared: make_tileview(B_shared, (M, 512), (-2, -1)),
                }
            )

            for i, j in T.Tiles([M, N], parallel=True):
                B_shared[i, j] = A_shared[i, j] + A_shared[i, j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[8, 512], expected_execution_domain_axes=[0, 1])


def test_manual_layout_bounded_rowmajor_rejects_shape_beyond_covered_shape():
    """A manual execution TileView cannot exceed the row-major envelope."""
    from tilelang.tileview import make_tileview

    M, N = 8, 128
    layout = make_aligned_row_major((M, N), "float16", align_bytes=1024)

    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            T.annotate_layout({A_shared: layout, B_shared: layout})
            T.annotate_tileview(
                {
                    A_shared: make_tileview(A_shared, (M, 1024), (-2, -1)),
                    B_shared: make_tileview(B_shared, (M, 1024), (-2, -1)),
                }
            )

            for i, j in T.Tiles([M, N], parallel=True):
                B_shared[i, j] = A_shared[i, j] + A_shared[i, j]

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with (
        tvm.target.Target(target),
        pytest.raises(
            tvm.error.InternalError,
            match="outside the layout-bounded execution envelope",
        ),
    ):
        apply_sunmmio_passes(mod, target)


def test_infer_tileview_3d_swapped_domain_binding():
    """Trailing 2D execution axes should be inferred independently of loop order."""
    K, M, N = 4, 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((K, M, N), "float16")
            B_shared = T.alloc_shared((K, M, N), "float16")
            C_shared = T.alloc_shared((K, M, N), "float16")

            T.copy(A[0:M, 0:N], A_shared[0, :, :])
            T.copy(B[0:M, 0:N], B_shared[0, :, :])

            for k, j, i in T.Tiles([K, N, M], parallel=True):
                C_shared[k, i, j] = A_shared[k, i, j] * B_shared[k, i, j]

            T.copy(C_shared[0, :, :], C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[2, 128], expected_execution_domain_axes=[2, 1])


def test_infer_tileview_1d_fp16_fills_register():
    """1D row-major fp16 buffers should use 256 contiguous elements."""
    N = 1024

    @T.prim_func
    def main(
        A: T.Tensor((N,), "float16"),
        B: T.Tensor((N,), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((N,), "float16")
            B_shared = T.alloc_shared((N,), "float16")

            T.copy(A[0:N], A_shared)

            for i in T.Tiles([N], parallel=True):
                B_shared[i] = A_shared[i] + A_shared[i]

            T.copy(B_shared, B[0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[256], expected_execution_domain_axes=[0])


def test_infer_tileview_2d_rowmajor_fp32():
    """Row-major fp32 buffers should use full-row tiles limited by 128 elements."""
    M, N = 64, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float32")
            B_shared = T.alloc_shared((M, N), "float32")
            C_shared = T.alloc_shared((M, N), "float32")

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[1, 128], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_2d_blockwise_fp32():
    """Blockwise fp32 buffers should use one full ZZ block."""
    M, N = 256, 128

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float32"),
        B: T.Tensor((M, N), "float32"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float32")
            B_shared = T.alloc_shared((M, N), "float32")
            C_shared = T.alloc_shared((M, N), "float32")

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                    B_shared: make_zz_layout(B_shared),
                    C_shared: make_zz_layout(C_shared),
                }
            )

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[32, 32], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_blockwise_small_height():
    """Blockwise candidates may exceed a small domain and rely on masks."""
    domain_M, buffer_M, N = 4, 32, 128

    @T.prim_func
    def main(
        A: T.Tensor((buffer_M, N), "float16"),
        B: T.Tensor((buffer_M, N), "float16"),
        C: T.Tensor((domain_M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((buffer_M, N), "float16")
            B_shared = T.alloc_shared((buffer_M, N), "float16")
            C_shared = T.alloc_shared((buffer_M, N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                    B_shared: make_zz_layout(B_shared),
                    C_shared: make_zz_layout(C_shared),
                }
            )

            T.copy(A[0:buffer_M, 0:N], A_shared)
            T.copy(B[0:buffer_M, 0:N], B_shared)

            for i, j in T.Tiles([domain_M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] * B_shared[i, j]

            T.copy(C_shared[0:domain_M, 0:N], C[0:domain_M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[32, 32], expected_execution_domain_axes=[0, 1])

    stores = collect_stores(mod["main"], "C_shared")
    assert stores, "Expected lowered C_shared stores"
    assert all(store.predicate is not None for store in stores)
    assert any("* 32" in str(store.indices[0]) or "*32" in str(store.indices[0]) for store in stores)
    assert all("< 4" in str(store.predicate) or "<4" in str(store.predicate) for store in stores)
    assert all("< 128" not in str(store.predicate) and "<128" not in str(store.predicate) for store in stores)
    assert not collect_if_conditions(mod["main"]), "Expected no full/tail branch when every tile is partial"


def test_infer_tileview_rowmajor_wide_row_uses_single_row_tile():
    """Wide row-major buffers should fall back to a single-row contiguous tile."""
    M, N = 64, 512

    @T.prim_func
    def main(
        A: T.Tensor((M, N), "float16"),
        B: T.Tensor((M, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((M, N), "float16")
            B_shared = T.alloc_shared((M, N), "float16")
            C_shared = T.alloc_shared((M, N), "float16")

            T.copy(A[0:M, 0:N], A_shared)
            T.copy(B[0:M, 0:N], B_shared)

            for i, j in T.Tiles([M, N], parallel=True):
                C_shared[i, j] = A_shared[i, j] + B_shared[i, j]

            T.copy(C_shared, C[0:M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[1, 256], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_rowmajor_region_height_offset():
    """Aligned row-major height offsets should preserve the best full-row tile."""
    src_M, dst_M, N = 64, 32, 128

    @T.prim_func
    def main(
        A: T.Tensor((src_M, N), "float16"),
        C: T.Tensor((dst_M, N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((src_M, N), "float16")
            C_shared = T.alloc_shared((dst_M, N), "float16")

            T.copy(A[0:src_M, 0:N], A_shared)

            for i, j in T.Tiles([dst_M, N], parallel=True):
                C_shared[i, j] = A_shared[i + 16, j] + A_shared[i + 16, j]

            T.copy(C_shared, C[0:dst_M, 0:N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[2, 128], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_rowmajor_region_width_offset():
    """Wide row-major regions should use the best single-row contiguous tile."""
    src_M, src_N = 32, 512
    dst_M, dst_N = 32, 256

    @T.prim_func
    def main(
        A: T.Tensor((src_M, src_N), "float16"),
        C: T.Tensor((dst_M, dst_N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((src_M, src_N), "float16")
            C_shared = T.alloc_shared((dst_M, dst_N), "float16")

            T.copy(A[0:src_M, 0:src_N], A_shared)

            for i, j in T.Tiles([dst_M, dst_N], parallel=True):
                C_shared[i, j] = A_shared[i, j + 256] + A_shared[i, j + 256]

            T.copy(C_shared, C[0:dst_M, 0:dst_N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tl.transform.LowerTilesLoop()(mod)
    assert_scope_plan(mod, expected_tile_size=[1, 256], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_blockwise_region_height_and_width_offset():
    """A half-block row offset selects the largest aligned block factor."""
    src_M, src_N = 64, 64
    dst_M, dst_N = 32, 32

    @T.prim_func
    def main(
        A: T.Tensor((src_M, src_N), "float16"),
        C: T.Tensor((dst_M, dst_N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((src_M, src_N), "float16")
            C_shared = T.alloc_shared((dst_M, dst_N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                    C_shared: make_zz_layout(C_shared),
                }
            )

            T.copy(A[0:src_M, 0:src_N], A_shared)

            for i, j in T.Tiles([dst_M, dst_N], parallel=True):
                C_shared[i, j] = A_shared[i + 16, j + 32] + A_shared[i + 16, j + 32]

            T.copy(C_shared, C[0:dst_M, 0:dst_N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_scope_plan(mod, expected_tile_size=[16, 32], expected_execution_domain_axes=[0, 1])


def test_infer_tileview_blockwise_region_misaligned_width_offset_falls_back():
    """Misaligned blockwise width offsets should fall back from 2D planning.

    With fp16 and 64-byte RSRAM alignment, the minimum tile width is 32
    elements.  Offset 16 is not aligned to 32, so no feasible tile exists.
    """
    src_M, src_N = 64, 64
    dst_M, dst_N = 32, 32

    @T.prim_func
    def main(
        A: T.Tensor((src_M, src_N), "float16"),
        C: T.Tensor((dst_M, dst_N), "float16"),
    ):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((src_M, src_N), "float16")
            C_shared = T.alloc_shared((dst_M, dst_N), "float16")

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                    C_shared: make_zz_layout(C_shared),
                }
            )

            T.copy(A[0:src_M, 0:src_N], A_shared)

            for i, j in T.Tiles([dst_M, dst_N], parallel=True):
                C_shared[i, j] = A_shared[i, j + 16] + A_shared[i, j + 16]

            T.copy(C_shared, C[0:dst_M, 0:dst_N])

    mod = IRModule.from_expr(main.with_attr("global_symbol", "main"))
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    scope_root, _ = collect_tile_annotations(mod["main"])
    assert scope_root is None or len(scope_root["tile.tile_size"]) == 1
