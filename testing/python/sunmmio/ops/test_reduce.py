import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.layout import make_zz_layout, make_aligned_row_major, make_row_major
from tilelang.layout.cute_layout import is_same_layout
from tilelang.tileview import make_tileview
from tilelang.utils.target import SUNMMIO_TARGET_DESC
from tvm.tir import Block
from tvm.tir.stmt_functor import post_order_visit
import pytest

tilelang.env.disable_cache()


def apply_sunmmio_passes(mod, target):
    """Apply the SUNMMIO pass pipeline used for Reduce lowering."""
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
    return mod


def assert_reduce_lowering_is_ssa(mod):
    """Reduce lowering should be SSA-clean before LowerOpaqueBlock/ConvertSSA."""
    assert tvm.tir.analysis.verify_ssa(mod["main"]), mod.script()


def _layout_map(mod):
    """Extract {buffer_name: layout} from a module's block layout_map
    annotations (the module having been lowered via apply_sunmmio_passes)."""
    result = {}

    def visit(node):
        if isinstance(node, Block) and "layout_map" in node.annotations:
            for buf, layout in node.annotations["layout_map"].items():
                result[buf.name] = layout

    post_order_visit(mod["main"].body, visit)
    return result


def _layout_logical_shape(layout):
    return tuple(int(x) for x in layout.logical_shape)


def _compute_layouts_for_shape(layouts, shape):
    """Return layouts for non-staging RSRAM compute buffers with this logical shape.

    LegalizeSunmmioDataPath rewrites user buffers such as Out_shared into
    compact compute buffers such as dst_buffer, plus row-major layout_stage
    buffers on the DRAM boundary.  These tests care about the compute layout,
    so skip the staging aliases.
    """
    target_shape = tuple(shape)
    return [
        (name, layout) for name, layout in layouts.items() if "layout_stage" not in name and _layout_logical_shape(layout) == target_shape
    ]


def _assert_compute_layout(layouts, shape, expected_layout):
    candidates = _compute_layouts_for_shape(layouts, shape)
    assert candidates, f"Missing compute layout for shape {shape}; layouts={list(layouts)}"
    assert any(is_same_layout(layout, expected_layout) for _, layout in candidates), (
        f"No compute layout for shape {shape} matches expected {expected_layout}; "
        f"candidates={[(name, layout) for name, layout in candidates]}"
    )


def _assert_no_compute_layout(layouts, shape, unexpected_layout):
    candidates = _compute_layouts_for_shape(layouts, shape)
    assert candidates, f"Missing compute layout for shape {shape}; layouts={list(layouts)}"
    assert not any(is_same_layout(layout, unexpected_layout) for _, layout in candidates), (
        f"Unexpected compute layout for shape {shape}: {unexpected_layout}; candidates={[(name, layout) for name, layout in candidates]}"
    )


def _collect_alloc_buffer_names(func):
    names = []

    def visit(node):
        if isinstance(node, Block):
            for buf in node.alloc_buffers:
                names.append(buf.name)

    post_order_visit(func.body, visit)
    return names


@tvm.tir.functor.visitor
class ReduceIRChecker(tvm.tir.PyStmtExprVisitor):
    def __init__(self, target_buffer_name="Out_shared"):
        super().__init__()
        self.target_buffer_name = target_buffer_name
        self.has_in_tile_reduce = False
        self.scope_root = None
        self.scope_entry_count = 0
        self.execution_axes = []
        self.interior_axes = []
        self.saw_legacy_stage = False
        self.saw_legacy_execution = False

    def visit_for_(self, op):
        ann = op.annotations
        if ann:
            if "tile.domain" in ann:
                self.scope_root = op
            if ann.get("tile.scope_entry", 0) == 1:
                self.scope_entry_count += 1
            if "tile.execution_axis" in ann:
                self.execution_axes.append(int(ann["tile.execution_axis"]))
            if ann.get("tile.interior", 0) == 1:
                self.interior_axes.append(int(ann["tile.interior_axis"]))
            if "tile.loop_stage" in ann:
                self.saw_legacy_stage = True
            if "tile.execution" in ann:
                self.saw_legacy_execution = True

        super().visit_for_(op)

    def visit_call_(self, op):
        if op.op.name == "tl.vector_core_in_tile_reduce":
            self.has_in_tile_reduce = True
        super().visit_call_(op)


def reduce_kernel_builder(shape, reduce_axis, dtype="float16"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:  # Handle scalar reduction case
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            # For Sunmmio, src and dst must be in shared.rsram for vector core operations
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis)
            T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


def reduce_kernel_with_blockwise_layout_builder(shape, reduce_axis, dtype="float32"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.annotate_layout(
                {
                    A_shared: make_zz_layout(A_shared),
                }
            )

            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=reduce_axis)
            T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


def apply_reduce_op(reduce_op, buffer, out, reduce_axis, clear=True):
    if reduce_op == "sum":
        T.reduce_sum(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "max":
        T.reduce_max(buffer, out, dim=reduce_axis, clear=clear)
    elif reduce_op == "min":
        T.reduce_min(buffer, out, dim=reduce_axis, clear=clear)
    else:
        raise ValueError(f"Unsupported reduce_op={reduce_op}")


def reduce_kernel_with_tileview_builder(shape, reduce_axis, tile_size=(8, 32), dtype="float16", clear=True, reduce_op="sum"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.annotate_tileview({A_shared: make_tileview(A_shared, tile_size, (-2, -1))})
            T.copy(A, A_shared)
            if not clear:
                T.copy(Out, Out_shared)
            apply_reduce_op(reduce_op, A_shared, Out_shared, reduce_axis, clear=clear)
            T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


def unaligned_reduce_kernel_builder(shape, reduce_axis, dtype="float16", clear=True, reduce_op="sum"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]
    input_boundary_layout = make_aligned_row_major(shape, dtype, 1024) if len(shape) == 1 else None
    if input_boundary_layout is not None:
        placement = T.placement.replicated()
        input_type = T.MeshTensor(shape, placement, dtype, layout=input_boundary_layout)
        output_type = T.Tensor(out_shape, dtype)
    else:
        input_type = T.Tensor(shape, dtype)
        output_type = T.Tensor(out_shape, dtype)

    @T.prim_func
    def main(A: input_type, Out: output_type):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            if input_boundary_layout is not None:
                T.annotate_layout({A_shared: input_boundary_layout})

            T.copy(A, A_shared)
            if not clear:
                if input_boundary_layout is not None:
                    # A rank-1 reduction produces an effective-rank-0 (1,) result,
                    # which is outside the aligned-row DMA carrier contract.
                    T.fill(Out_shared, 0)
                else:
                    T.copy(Out, Out_shared)
            apply_reduce_op(reduce_op, A_shared, Out_shared, reduce_axis, clear=clear)
            if input_boundary_layout is None:
                T.copy(Out_shared, Out)

    return tvm.IRModule({"main": main})


def multi_reduce_kernel_builder(shape=(32, 128, 128), reduce_axis=1, dtype="float16"):
    out_shape = list(shape[:reduce_axis]) + list(shape[reduce_axis + 1 :])
    if not out_shape:
        out_shape = [1]

    @T.prim_func
    def main(A: T.Tensor(shape, dtype), Out0: T.Tensor(out_shape, dtype), Out1: T.Tensor(out_shape, dtype)):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared(shape, dtype, scope="shared.rsram")
            Out0_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")
            Out1_shared = T.alloc_shared(out_shape, dtype, scope="shared.rsram")

            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out0_shared, dim=reduce_axis)
            T.reduce_sum(A_shared, Out1_shared, dim=reduce_axis)
            T.copy(Out0_shared, Out0)
            T.copy(Out1_shared, Out1)

    return tvm.IRModule({"main": main})


def _collect_buffer_loads(func, buffer_name):
    loads = []

    def visitor(node):
        if isinstance(node, tvm.tir.BufferLoad) and node.buffer.name == buffer_name:
            loads.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    return loads


def _collect_buffer_stores(func, buffer_name):
    stores = []

    def visitor(node):
        if isinstance(node, tvm.tir.BufferStore) and node.buffer.name == buffer_name:
            stores.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    return stores


def _collect_buffer_loads_any(func, buffer_names):
    loads = []
    buffer_names = set(buffer_names)

    def visitor(node):
        if isinstance(node, tvm.tir.BufferLoad) and node.buffer.name in buffer_names:
            loads.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    return loads


def _collect_buffer_stores_any(func, buffer_names):
    stores = []
    buffer_names = set(buffer_names)

    def visitor(node):
        if isinstance(node, tvm.tir.BufferStore) and node.buffer.name in buffer_names:
            stores.append(node)

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    return stores


def _collect_tile_loop_extents(func):
    extents = []

    @tvm.tir.functor.visitor
    class TileLoopVisitor(tvm.tir.PyStmtExprVisitor):
        def visit_for_(self, op):
            if op.annotations and "tile.domain" in op.annotations or op.annotations and "tile.execution_axis" in op.annotations:
                extents.append(int(op.extent))
            super().visit_for_(op)

    TileLoopVisitor().visit_stmt(func.body)
    return extents


def _collect_tile_domain_roots(func):
    roots = []

    @tvm.tir.functor.visitor
    class TileDomainVisitor(tvm.tir.PyStmtExprVisitor):
        def visit_for_(self, op):
            if op.annotations and "tile.domain" in op.annotations:
                roots.append(op)
            super().visit_for_(op)

    TileDomainVisitor().visit_stmt(func.body)
    return roots


def _tile_domain(loop):
    return [int(x) for x in loop.annotations["tile.domain"]]


def _collect_execution_loop_extents(root):
    """Collect execution loops under one tile.domain, ignoring nested domains."""
    execution_extents = []

    @tvm.tir.functor.visitor
    class ExecutionLoopVisitor(tvm.tir.PyStmtExprVisitor):
        def __init__(self):
            super().__init__()
            self.depth = 0

        def visit_for_(self, op):
            ann = op.annotations
            if self.depth > 0 and ann and "tile.domain" in ann:
                return
            if ann and "tile.execution_axis" in ann:
                axis = int(ann["tile.execution_axis"])
                execution_extents.append((axis, int(op.extent)))
            self.depth += 1
            super().visit_for_(op)
            self.depth -= 1

    ExecutionLoopVisitor().visit_stmt(root)
    return execution_extents


def _get_lowered_reduce_dst_layout(layouts, fallback_name="Out_shared"):
    if "dst_buffer" in layouts:
        return layouts["dst_buffer"]
    return layouts[fallback_name]


def _get_lowered_reduce_src_layout(layouts, fallback_name):
    if "src_buffer" in layouts:
        return layouts["src_buffer"]
    return layouts[fallback_name]


def _ceildiv(lhs, rhs):
    return (lhs + rhs - 1) // rhs


def _expected_tiled_reduce(shape, reduce_axis):
    return len(shape) == 1 or reduce_axis >= len(shape) - 2


def _tail_identity_text(reduce_op):
    if reduce_op == "sum":
        return "T.float16(0.0)"
    if reduce_op == "max":
        return 'T.float16("-inf")'
    if reduce_op == "min":
        return 'T.float16("inf")'
    raise ValueError(f"Unsupported reduce_op={reduce_op}")


# (Shape, ReduceAxis, ExpectedInTileReduce)
# For Sunmmio, all dimensions should be multiples of 32 for simplicity in these tests.
REDUCE_TEST_CASES = [
    ((1024,), 0, True),
    ((32, 1024), 1, True),
    # 2D
    ((128, 128), 1, True),
    ((128, 128), 0, True),
    # 3D
    ((32, 128, 128), 2, True),
    ((32, 128, 128), 1, True),
    ((32, 128, 128), 0, False),
    # 4D
    ((32, 32, 128, 128), 3, True),
    ((32, 32, 128, 128), 1, False),
    # 5D
    ((32, 32, 32, 128, 128), 4, True),
    ((32, 32, 32, 128, 128), 0, False),
]


@pytest.mark.parametrize("shape, reduce_axis, expected_in_tile", REDUCE_TEST_CASES)
def test_tilelang_reduce_sunmmio(shape, reduce_axis, expected_in_tile):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_builder(shape, reduce_axis)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)

    assert checker.scope_root is not None, "Missing tile.domain root on lowered reduction"
    root_ann = checker.scope_root.annotations
    tile_size = [int(x) for x in root_ann["tile.tile_size"]]
    execution_domain_axes = [int(x) for x in root_ann["tile.execution_domain_axes"]]

    if expected_in_tile:
        assert checker.has_in_tile_reduce, "Expected vector_core_in_tile_reduce intrinsic but not found"
    else:
        assert not checker.has_in_tile_reduce, "Did not expect vector_core_in_tile_reduce intrinsic but found it"

    assert checker.scope_entry_count == 1, "Expected exactly one tile.scope_entry annotation"
    assert not checker.saw_legacy_stage, "Reduction should not emit legacy tile.loop_stage annotations"
    assert not checker.saw_legacy_execution, "Reduction should not emit legacy tile.execution annotations"
    assert sorted(checker.execution_axes) == list(range(len(tile_size))), (
        "tile.execution_axis annotations should cover every execution axis"
    )
    assert len(execution_domain_axes) == len(tile_size), "tile.execution_domain_axes rank must match tile.tile_size"
    assert set(checker.interior_axes).issuperset(set(range(len(tile_size)))), "Missing tile.interior annotations for one or more tile axes"


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_tilelang_reduce_sunmmio_uses_layout_bounded_zz_source_tile(dtype):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_with_blockwise_layout_builder((128, 128), 1, dtype=dtype)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)

    assert checker.scope_root is not None, "Missing tile.domain root on lowered reduction"
    root_ann = checker.scope_root.annotations
    tile_size = [int(x) for x in root_ann["tile.tile_size"]]
    execution_domain_axes = [int(x) for x in root_ann["tile.execution_domain_axes"]]

    assert checker.has_in_tile_reduce, "Expected vector_core_in_tile_reduce intrinsic but not found"
    assert tile_size == [32, 32]
    assert execution_domain_axes == [0, 1]


def test_tilelang_reduce_sunmmio_bounds_source_tile_by_destination_layout():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_with_blockwise_layout_builder((4, 32), 1, dtype="float32")

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)
    assert checker.scope_root is not None
    assert [int(x) for x in checker.scope_root.annotations["tile.tile_size"]] == [16, 32]


def test_tilelang_reduce_sunmmio_manual_full_zz_block_source_tile():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_with_tileview_builder((128, 128), reduce_axis=1, tile_size=(32, 32))

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)
    assert checker.scope_root is not None
    assert [int(x) for x in checker.scope_root.annotations["tile.tile_size"]] == [32, 32]


def test_tilelang_reduce_sunmmio_rejects_incompatible_manual_destination_tileview():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((32, 128), "float32"), Out: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((32, 128), "float32", scope="shared.rsram")
            Out_shared = T.alloc_shared((32,), "float32", scope="shared.rsram")
            T.annotate_tileview({Out_shared: make_tileview(Out_shared, (64,), (-1,))})
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=1)
            T.copy(Out_shared, Out)

    with (
        tvm.target.Target(target),
        pytest.raises(
            tvm.error.InternalError,
            match="Cannot infer a legal Sunmmio reduction TileView plan",
        ),
    ):
        apply_sunmmio_passes(tvm.IRModule({"main": main}), target)


def test_tilelang_reduce_sunmmio_uses_row_major_covered_source_tile():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((1000,), "float16"), Out: T.Tensor((1,), "float16")):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((1000,), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((1,), "float16", scope="shared.rsram")
            T.annotate_layout({A_shared: make_aligned_row_major((1000,), "float16", 64)})
            T.annotate_tileview({A_shared: make_tileview(A_shared, (1024,), (-1,))})
            T.copy(A, A_shared)
            T.reduce_max(A_shared, Out_shared, dim=0)
            T.copy(Out_shared, Out)

    mod = tvm.IRModule({"main": main})

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)
    assert checker.scope_root is not None
    assert [int(x) for x in checker.scope_root.annotations["tile.tile_size"]] == [1024]
    script = mod.script()
    assert 'T.float16("-inf")' in script
    assert "< 1000" in script or "<1000" in script


def test_tilelang_reduce_sunmmio_multiple_reduces_are_ssa_clean():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = multi_reduce_kernel_builder()

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)

    assert_reduce_lowering_is_ssa(mod)


def test_tilelang_reduce_sunmmio_tiled_axis_tail_uses_if_then_else_mask():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_with_tileview_builder((8, 63, 250), reduce_axis=2)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    func = mod["main"]
    tile_loop_extents = _collect_tile_loop_extents(func)
    assert tile_loop_extents.count(8) >= 3, "Expected ceildiv tile extents for dimensions 8, 63, and 250"
    assert 7 not in tile_loop_extents, "truncdiv(63, 8) would drop the spatial tail tile"

    a_load_predicates = [load.predicate for load in _collect_buffer_loads(func, "A_shared") if load.predicate is not None]
    assert not a_load_predicates, "Reduce-axis tail masking should be expressed by if_then_else, not BufferLoad.predicate"
    script = mod.script()
    assert "T.if_then_else" in script
    assert "T.float16(0.0)" in script
    assert "< 250" in script or "<250" in script
    assert "predicate=" not in script

    out_store_predicates = [store.predicate for store in _collect_buffer_stores(func, "Out_shared") if store.predicate is not None]
    assert not out_store_predicates, "Reduce final write-back should remain unpredicated"


@pytest.mark.parametrize(
    "reduce_op",
    [
        "max",
        "min",
    ],
)
def test_tilelang_reduce_sunmmio_tiled_axis_tail_uses_predicated_update(reduce_op):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_with_tileview_builder((8, 63, 250), reduce_axis=2, reduce_op=reduce_op)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    script = mod.script()
    assert "T.if_then_else" in script
    expected_identity = 'T.float16("-inf")' if reduce_op == "max" else 'T.float16("inf")'
    assert expected_identity in script
    a_load_predicates = [load.predicate for load in _collect_buffer_loads(mod["main"], "A_shared") if load.predicate is not None]
    acc_store_predicates = [
        store.predicate for store in _collect_buffer_stores(mod["main"], "Out_shared_acc") if store.predicate is not None
    ]
    assert not a_load_predicates
    assert not acc_store_predicates
    assert "< 250" in script or "<250" in script


UNALIGNED_REDUCE_CASES = [
    ((1000,), 0, True),
    ((1000,), 0, False),
    ((33, 50), 0, True),
    ((33, 50), 0, False),
    ((33, 50), 1, True),
    ((33, 50), 1, False),
    ((5, 43, 249), 0, True),
    ((5, 43, 249), 0, False),
    ((5, 43, 249), 1, True),
    ((5, 43, 249), 1, False),
    ((5, 43, 249), 2, True),
    ((5, 43, 249), 2, False),
]


@pytest.mark.parametrize("shape, reduce_axis, clear", UNALIGNED_REDUCE_CASES)
def test_tilelang_reduce_sunmmio_unaligned_cases_from_tir_dump(shape, reduce_axis, clear):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = unaligned_reduce_kernel_builder(shape, reduce_axis=reduce_axis, clear=clear)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    func = mod["main"]
    checker = ReduceIRChecker()
    checker.visit_stmt(func.body)
    assert checker.scope_root is not None, "Missing tile.domain root on lowered unaligned reduction"
    assert checker.has_in_tile_reduce == _expected_tiled_reduce(shape, reduce_axis)

    roots = [root for root in _collect_tile_domain_roots(func) if _tile_domain(root) == list(shape)]
    assert len(roots) == 1
    root = roots[0]
    tile_size = [int(x) for x in root.annotations["tile.tile_size"]]
    execution_domain_axes = [int(x) for x in root.annotations["tile.execution_domain_axes"]]
    domain = _tile_domain(root)
    assert domain == list(shape)
    assert len(tile_size) == len(execution_domain_axes)

    execution_extents = _collect_execution_loop_extents(root)
    assert len(execution_extents) == len(tile_size)
    for axis, extent in execution_extents:
        domain_axis = execution_domain_axes[axis]
        expected_extent = _ceildiv(shape[domain_axis], tile_size[axis])
        assert extent == expected_extent
        if shape[domain_axis] % tile_size[axis] != 0:
            assert extent > shape[domain_axis] // tile_size[axis]

    script = mod.script()
    a_load_predicates = [
        load.predicate for load in _collect_buffer_loads_any(func, ("A_shared", "src_buffer")) if load.predicate is not None
    ]
    acc_store_predicates = [
        store.predicate for store in _collect_buffer_stores_any(func, ("Out_shared_acc", "dst_buffer_acc")) if store.predicate is not None
    ]
    assert not acc_store_predicates

    is_reduce_axis_tiled = reduce_axis in execution_domain_axes
    reduce_tile_size = tile_size[execution_domain_axes.index(reduce_axis)] if is_reduce_axis_tiled else 1
    has_reduce_axis_tail = is_reduce_axis_tiled and shape[reduce_axis] % reduce_tile_size != 0
    if has_reduce_axis_tail:
        assert not a_load_predicates
        assert "T.if_then_else" in script
        assert _tail_identity_text("sum") in script
        assert f"< {shape[reduce_axis]}" in script or f"<{shape[reduce_axis]}" in script
    else:
        assert not a_load_predicates

    if is_reduce_axis_tiled:
        has_reduce_result_buffer = any(name.endswith("_res") for name in _collect_alloc_buffer_names(func))
        assert has_reduce_result_buffer == (not clear)


@pytest.mark.parametrize("reduce_op", ["sum", "max", "min"])
def test_tilelang_reduce_sunmmio_unaligned_tiled_axis_tail_identity(reduce_op):
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = unaligned_reduce_kernel_builder((5, 43, 249), reduce_axis=2, clear=True, reduce_op=reduce_op)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    script = mod.script()
    assert "T.if_then_else" in script
    assert _tail_identity_text(reduce_op) in script
    assert "i2 * 32 + kj < 249" in script
    assert "predicate=" not in script


def test_tilelang_reduce_sunmmio_non_tiled_axis_tail_has_no_reduce_predicate():
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)
    mod = reduce_kernel_with_tileview_builder((8, 63, 250), reduce_axis=0)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(mod, target)
    assert_reduce_lowering_is_ssa(mod)

    checker = ReduceIRChecker()
    checker.visit_stmt(mod["main"].body)
    assert not checker.has_in_tile_reduce

    a_load_predicates = [load.predicate for load in _collect_buffer_loads(mod["main"], "A_shared") if load.predicate is not None]
    out_store_predicates = [store.predicate for store in _collect_buffer_stores(mod["main"], "Out_shared") if store.predicate is not None]
    assert not a_load_predicates
    assert not out_store_predicates


def test_tilelang_reduce_sunmmio_blocked_axis_yields_aligned_rowmajor():
    """Reducing a blocked (ZZ) axis to a non-32-multiple output gives the dst an
    alignment-padded row-major layout (e.g. (40,) -> covered 64), not plain."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((64, 40), "float16"), Out: T.Tensor((40,), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((64, 40), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((40,), "float16", scope="shared.rsram")
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=0)
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(tvm.IRModule({"main": main}), target)
    layouts = _layout_map(mod)

    _assert_compute_layout(layouts, (40,), make_aligned_row_major((40,), "float16", 64))
    _assert_no_compute_layout(layouts, (40,), make_row_major((40,)))


def test_tilelang_reduce_sunmmio_3d_blocked_axis_aligned():
    """3D ZZ source, reduce the inner blocked axis -> 2D aligned row-major dst
    with a non-32-multiple inner extent (the (2,40,256) -> (2,40) shape)."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((2, 40, 256), "float16"), Out: T.Tensor((2, 40), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((2, 40, 256), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((2, 40), "float16", scope="shared.rsram")
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=2)
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(tvm.IRModule({"main": main}), target)
    layouts = _layout_map(mod)
    _assert_compute_layout(layouts, (2, 40), make_aligned_row_major((2, 40), "float16", 64))


def test_tilelang_reduce_sunmmio_chained_reduce_stays_aligned():
    """Reducing twice: the first reduce makes an (8,40) aligned row-major; the
    second reduce off that *unblocked* buffer must stay aligned (the chained
    path goes through DeriveLayoutLike -> MakeAlignedRowMajor, not plain)."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((8, 64, 40), "float16"), Out: T.Tensor((40,), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((8, 64, 40), "float16", scope="shared.rsram")
            M_shared = T.alloc_shared((8, 40), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((40,), "float16", scope="shared.rsram")
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, M_shared, dim=1)  # blocked axis -> aligned (8,40)
            T.reduce_sum(M_shared, Out_shared, dim=0)  # off unblocked -> aligned
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(tvm.IRModule({"main": main}), target)
    layouts = _layout_map(mod)
    _assert_compute_layout(layouts, (8, 40), make_aligned_row_major((8, 40), "float16", 64))
    _assert_compute_layout(layouts, (40,), make_aligned_row_major((40,), "float16", 64))
    _assert_no_compute_layout(layouts, (40,), make_row_major((40,)))


def test_tilelang_reduce_sunmmio_nonblocked_reduce_preserves_zz():
    """Reducing a NON-blocked leading axis keeps the surviving ZZ block
    structure (DeriveLayoutLike path), not a flat row-major."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((40, 64, 64), "float16"), Out: T.Tensor((64, 64), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((40, 64, 64), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((64, 64), "float16", scope="shared.rsram")
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=0)
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(tvm.IRModule({"main": main}), target)
    layouts = _layout_map(mod)
    _assert_compute_layout(layouts, (64, 64), make_zz_layout((64, 64), [0, 1], (32, 32)))
    _assert_no_compute_layout(layouts, (64, 64), make_row_major((64, 64)))


def test_tilelang_reduce_sunmmio_aligned_dst_is_noop_when_32_multiple():
    """A 32-multiple reduce output gets no padding: aligned row-major collapses
    to plain row-major (no spurious covered-extent inflation)."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((64, 64), "float16"), Out: T.Tensor((64,), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((64, 64), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((64,), "float16", scope="shared.rsram")
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=0)
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(tvm.IRModule({"main": main}), target)
    layouts = _layout_map(mod)
    _assert_compute_layout(layouts, (64,), make_row_major((64,)))


def test_tilelang_reduce_sunmmio_aligned_output_lowers_end_to_end():
    """End-to-end (through LowerTileOp): reduce dim1 of a ZZ (40,64) source — a
    blocked axis with a 32-multiple inner extent — yields an aligned, covered-
    padded (40,) output that lowers and stores to unpadded DRAM via an unpad
    transform.  Two sunmmio_layout_transforms: ZZ-reblock on the load, unpad on
    the store.  This proves the aligned (covered != logical) reduce dst is fully
    lowerable, not just inferred."""
    target = tvm.target.Target(SUNMMIO_TARGET_DESC)

    @T.prim_func
    def main(A: T.Tensor((40, 64), "float16"), Out: T.Tensor((40,), "float16")):
        with T.Kernel(1, threads=128) as (bx,):
            A_shared = T.alloc_shared((40, 64), "float16", scope="shared.rsram")
            Out_shared = T.alloc_shared((40,), "float16", scope="shared.rsram")
            T.copy(A, A_shared)
            T.reduce_sum(A_shared, Out_shared, dim=1)
            T.copy(Out_shared, Out)

    with tvm.target.Target(target):
        mod = apply_sunmmio_passes(tvm.IRModule({"main": main}), target)
    layouts = _layout_map(mod)
    _assert_compute_layout(layouts, (40,), make_aligned_row_major((40,), "float16", 64))

    names = []
    post_order_visit(
        mod["main"].body,
        lambda n: names.append(n.op.name) if isinstance(n, tvm.tir.Call) and hasattr(n.op, "name") else None,
    )
    assert names.count("tl.sunmmio_layout_transform") == 2, names


if __name__ == "__main__":
    pytest.main([__file__])
