from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tilelang.layout import Layout
import tilelang.testing
from tvm.tir.stmt_functor import post_order_visit

_MVB_ATTR_KEYS = frozenset(
    [
        "tl.pipeline_mvb_num_stages",
        "tl.pipeline_mvb_stage_expr",
        "tl.pipeline_mvb_parity_expr",
        "tl.pipeline_context_num_stages",
    ]
)


@tvm.tir.transform.prim_func_pass(opt_level=0)
def _strip_mvb_attrs(func, mod, ctx):
    """Remove intermediate MVB attributes that are consumed by later passes."""

    def _visit(stmt):
        if isinstance(stmt, tvm.tir.AttrStmt) and str(stmt.attr_key) in _MVB_ATTR_KEYS:
            return stmt.body
        return None

    return func.with_body(tvm.tir.stmt_functor.ir_transform(func.body, None, _visit, ["tir.AttrStmt"]))


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = _strip_mvb_attrs(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"), True)


def _count_attrs_and_calls(func):
    attr_count = {}
    call_count = {}

    def _visit(node):
        if isinstance(node, tvm.tir.AttrStmt):
            key = str(node.attr_key)
            attr_count[key] = attr_count.get(key, 0) + 1
        elif isinstance(node, tvm.tir.Call) and isinstance(node.op, tvm.ir.Op):
            key = str(node.op.name)
            call_count[key] = call_count.get(key, 0) + 1

    post_order_visit(func.body, _visit)
    return attr_count, call_count


def _collect_attr_values(func, attr_key):
    values = []
    stmt = func.body if hasattr(func, "body") else func

    def _visit(node):
        if isinstance(node, tvm.tir.AttrStmt) and str(node.attr_key) == attr_key:
            value = node.value
            if isinstance(value, tvm.tir.IntImm):
                values.append(int(value.value))

    post_order_visit(stmt, _visit)
    return values


def _collect_attr_value_nodes(func, attr_key):
    values = []

    def _visit(node):
        if isinstance(node, tvm.tir.AttrStmt) and str(node.attr_key) == attr_key:
            values.append(node.value)

    post_order_visit(func.body, _visit)
    return values


def _collect_wait_args(func):
    wait_args = []
    stmt = func.body if hasattr(func, "body") else func

    def _visit(node):
        if (
            isinstance(node, tvm.tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and str(node.op.name) == "tir.ptx_wait_group"
            and len(node.args) == 1
        ):
            arg = node.args[0]
            if isinstance(arg, tvm.tir.IntImm):
                wait_args.append(int(arg.value))

    post_order_visit(stmt, _visit)
    return wait_args


def _find_pipelined_loop(func):
    loops = []

    def _visit(node):
        if isinstance(node, tvm.tir.For) and "tl_pipelined_num_stages" in node.annotations:
            loops.append(node)

    post_order_visit(func.body, _visit)
    assert loops, "Expected at least one loop annotated with tl_pipelined_num_stages"
    return loops[0]


def _count_copy_calls_with_annotation(func, annotation_key):
    annotated = 0
    total = 0

    def _visit(node):
        nonlocal annotated, total
        if not isinstance(node, tvm.tir.Call) or not isinstance(node.op, tvm.ir.Op):
            return
        if str(node.op.name) not in {"tl.tileop.copy", "tl.tileop.async_copy"}:
            return
        total += 1
        value = node.annotations.get(annotation_key) if node.annotations else None
        if isinstance(value, tvm.tir.IntImm) and int(value.value) != 0:
            annotated += 1

    post_order_visit(func.body, _visit)
    return annotated, total


def _find_block_with_layout_map(func):
    blocks = []

    def _visit(node):
        if isinstance(node, tvm.tir.Block) and "layout_map" in node.annotations:
            blocks.append(node)

    post_order_visit(func.body, _visit)
    assert blocks, "Expected at least one block with layout_map"
    return blocks[0]


def test_trival_pipeline():
    @T.prim_func
    def before(A: T.Tensor((16, 1), T.float32), C: T.Tensor((16, 1), T.float32)):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(0, 1, annotations={"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]}):
                with T.block():
                    T.reads(A[tx, i])
                    T.writes(C[tx, i])
                    B = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
                    with T.block():
                        T.reads(A[tx, i])
                        T.writes(B[tx, 0])
                        B[tx, 0] = A[tx, i] * T.float32(2)
                    with T.block():
                        T.reads(B[tx, 0])
                        T.writes(C[tx, i])
                        C[tx, i] = B[tx, 0] + T.float32(1)

    @T.prim_func
    def expected(A_handle: T.handle, C_handle: T.handle):
        A = T.match_buffer(A_handle, (16, 1), strides=(1, 1))
        C = T.match_buffer(C_handle, (16, 1), strides=(1, 1))
        tx = T.launch_thread("threadIdx.x", 16)
        B = T.decl_buffer((2, 16, 1), scope="shared")
        B[0, tx, 0] = A[tx, 0] * T.float32(2.0)
        C[tx, 0] = B[0, tx, 0] + T.float32(1.0)

    _check(before, expected)


def test_preserve_inline_cp_async_sync_in_pipeline_stage():
    @T.prim_func
    def before(A: T.Tensor((16,), T.uint8), B: T.Tensor((16,), T.uint8)):
        S = T.alloc_buffer((16,), dtype=T.uint8, scope="shared")
        for i in T.serial(
            4,
            annotations={
                "software_pipeline_stage": [T.int32(0), T.int32(1)],
                "software_pipeline_order": [T.int32(0), T.int32(1)],
                "software_pipeline_async_stages": [T.int32(0)],
            },
        ):
            with T.block():
                T.reads(A[i * 4 : i * 4 + 4])
                T.writes(S[i * 4 : i * 4 + 4])
                T.ptx_cp_async(
                    T.access_ptr(S[i * 4], "w", 4),
                    T.access_ptr(A[i * 4], "r", 4),
                    4,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
            with T.block():
                T.reads(S[i * 4 : i * 4 + 4])
                T.writes(B[i * 4 : i * 4 + 4])
                B[i * 4] = S[i * 4]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    mod = tl.transform.Simplify()(mod)

    attrs, calls = _count_attrs_and_calls(mod["main"])
    assert attrs.get("async_scope", 0) == 0
    assert attrs.get("async_commit_queue_scope", 0) == 0
    assert attrs.get("async_wait_queue_scope", 0) == 0
    assert attrs.get("async_wait_inflight_count", 0) == 0
    # Inline sync calls should remain explicit in the rewritten pipeline.
    assert calls.get("tir.ptx_commit_group", 0) > 0
    assert calls.get("tir.ptx_wait_group", 0) > 0


def test_async_pipeline_groups_multiple_copy_producers():
    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.float32),
        B: T.Tensor((16, 16), T.float32),
        C: T.Tensor((16, 16), T.float32),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                4,
                annotations={
                    "software_pipeline_stage": [0, 0, 1],
                    "software_pipeline_order": [0, 1, 2],
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_async_producers": [1, 1, 0],
                    "software_pipeline_async_producer_groups": [0, 0, -1],
                },
            ):
                with T.block("compute"):
                    T.reads(A[tx, i], B[tx, i])
                    T.writes(C[tx, i])
                    A_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
                    B_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
                    with T.block("copy_a"):
                        T.reads(A[tx, i])
                        T.writes(A_shared[tx, 0])
                        A_shared[tx, 0] = A[tx, i]
                    with T.block("copy_b"):
                        T.reads(B[tx, i])
                        T.writes(B_shared[tx, 0])
                        B_shared[tx, 0] = B[tx, i]
                    with T.block("consume"):
                        T.reads(A_shared[tx, 0], B_shared[tx, 0])
                        T.writes(C[tx, i])
                        C[tx, i] = A_shared[tx, 0] + B_shared[tx, 0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    mod = tl.transform.Simplify()(mod)

    attrs, calls = _count_attrs_and_calls(mod["main"])
    assert attrs.get("async_scope", 0) > 0
    assert attrs.get("async_commit_queue_scope", 0) == 0
    assert attrs.get("async_wait_queue_scope", 0) == 0
    assert attrs.get("async_wait_inflight_count", 0) == 0
    assert calls.get("tir.ptx_commit_group", 0) > 0
    assert 1 in _collect_wait_args(mod["main"])


def test_async_pipeline_only_wraps_producer_statements_from_explicit_group_annotations():
    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.float32),
        B: T.Tensor((16, 16), T.float32),
        C: T.Tensor((16, 16), T.float32),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                4,
                annotations={
                    "software_pipeline_stage": [0, 0, 0, 1],
                    "software_pipeline_order": [0, 1, 2, 3],
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_async_producers": [0, 1, 1, 0],
                    "software_pipeline_async_producer_groups": [-1, 0, 0, -1],
                },
            ):
                with T.block("compute"):
                    T.reads(A[tx, i], B[tx, i])
                    T.writes(C[tx, i])
                    A_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
                    B_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
                    with T.block("fill"):
                        T.reads()
                        T.writes(A_shared[tx, 0])
                        A_shared[tx, 0] = T.float32(0)
                    with T.block("copy_a"):
                        T.reads(A[tx, i])
                        T.writes(A_shared[tx, 0])
                        A_shared[tx, 0] = A[tx, i]
                    with T.block("copy_b"):
                        T.reads(B[tx, i])
                        T.writes(B_shared[tx, 0])
                        B_shared[tx, 0] = B[tx, i]
                    with T.block("consume"):
                        T.reads(A_shared[tx, 0], B_shared[tx, 0])
                        T.writes(C[tx, i])
                        C[tx, i] = A_shared[tx, 0] + B_shared[tx, 0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    mod = tl.transform.Simplify()(mod)

    attrs, calls = _count_attrs_and_calls(mod["main"])
    # Dead prologue/epilogue producer clones are now dropped during injection,
    # so only the live producer copies remain wrapped.
    assert attrs.get("async_scope", 0) == 4
    assert attrs.get("async_commit_queue_scope", 0) == 0
    assert calls.get("tir.ptx_commit_group", 0) == 2


def test_async_pipeline_marks_copy_ops_for_pipeline_managed_cp_async_sync():
    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.float32),
        B: T.Tensor((16, 16), T.float32),
        C: T.Tensor((16, 16), T.float32),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                4,
                annotations={
                    "software_pipeline_stage": [0, 0, 1],
                    "software_pipeline_order": [0, 1, 2],
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_async_producers": [1, 1, 0],
                    "software_pipeline_async_producer_groups": [0, 0, -1],
                },
            ):
                with T.block("compute"):
                    T.reads(A[tx, i], B[tx, i])
                    T.writes(C[tx, i])
                    A_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
                    B_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
                    T.copy(A[tx, i : i + 1], A_shared[tx, 0:1])
                    T.copy(B[tx, i : i + 1], B_shared[tx, 0:1])
                    C[tx, i] = A_shared[tx, 0] + B_shared[tx, 0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    annotated, total = _count_copy_calls_with_annotation(mod["main"], "no_implicit_async_commit_wait")
    assert total > 0
    assert annotated == total


def test_async_pipeline_does_not_mark_non_cp_async_compatible_copy():
    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.bfloat16),
        C: T.Tensor((16, 16), T.float32),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                4,
                annotations={
                    "software_pipeline_stage": [0, 1],
                    "software_pipeline_order": [0, 1],
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_async_producers": [1, 0],
                    "software_pipeline_async_producer_groups": [0, -1],
                },
            ):
                with T.block("compute"):
                    T.reads(A[tx, i])
                    T.writes(C[tx, i])
                    S = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
                    T.copy(A[tx, i : i + 1], S[tx, 0:1])
                    C[tx, i] = S[tx, 0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    annotated, total = _count_copy_calls_with_annotation(mod["main"], "no_implicit_async_commit_wait")
    assert total > 0
    assert annotated == 0


def test_async_pipeline_relaxes_loop_wait_and_splits_trailing_drain():
    @T.prim_func
    def before(A: T.Tensor((32,), T.uint8), B: T.Tensor((32,), T.uint8)):
        S = T.alloc_buffer((4,), dtype=T.uint8, scope="shared")
        for i in T.serial(
            0,
            4,
            annotations={
                "software_pipeline_stage": [0, 2],
                "software_pipeline_order": [0, 1],
                "software_pipeline_async_stages": [0],
                "software_pipeline_async_producers": [1, 0],
                "software_pipeline_async_producer_groups": [0, -1],
            },
        ):
            with T.block("copy"):
                T.reads(A[i * 4 : i * 4 + 4])
                T.writes(S[0:4])
                T.copy(A[i * 4 : i * 4 + 4], S[0:4])
            with T.block("consume"):
                T.reads(S[0:4])
                T.writes(B[i * 4 : i * 4 + 4])
                for j in range(4):
                    B[i * 4 + j] = S[j]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)

    func = mod["main"]
    loop = _find_pipelined_loop(func)
    loop_waits = _collect_wait_args(loop.body)
    all_waits = _collect_wait_args(func)

    assert loop_waits == [2], f"Expected relaxed loop wait to keep two groups in flight, got {loop_waits}"
    assert all_waits == [2, 2, 0], f"Expected trailing waits to split into retain+drain, got {all_waits}"


def test_degenerate_pipeline_with_single_stage_is_not_expanded():
    @T.prim_func
    def before(B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128) as _:
            frag = T.alloc_fragment((4, 128), T.float16)
            split = T.alloc_fragment((128,), T.float32)
            scale = T.alloc_fragment((128,), T.float32)
            for k in T.serial(
                4,
                annotations={"software_pipeline_stage": [2, 2], "software_pipeline_order": [0, 1], "tl_pipelined_num_stages": 2},
            ):
                for i in T.Parallel(128):
                    split[i] = T.Cast("float32", frag[k, i])
                for i in T.Parallel(128):
                    scale[i] = split[i]
                    B[i] = scale[i]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)

    func = mod["main"]
    attrs, calls = _count_attrs_and_calls(func)
    assert attrs.get("tl.pipeline_context_num_stages", 0) == 0
    assert attrs.get("tl.pipeline_mvb_num_stages", 0) == 0
    assert attrs.get("tl.pipeline_mvb_stage_expr", 0) == 0
    assert attrs.get("tl.pipeline_mvb_parity_expr", 0) == 0
    assert calls.get("tir.ptx_wait_group", 0) == 0
    assert "tl_pipelined_num_stages" not in func.script()
    assert "frag[k, i]" in func.script()
    assert "frag[2, i]" not in func.script()


def test_inject_software_pipeline_expands_annotated_layout():
    layout = Layout([8, 16], lambda i, j: i * 16 + j)

    @T.prim_func
    def before(A: T.Tensor((4, 8, 16), T.float16), B: T.Tensor((4, 8, 16), T.float16)):
        with T.block("root"):
            shared = T.alloc_buffer((8, 16), T.float16, scope="shared.dyn")
            T.annotate_layout({shared: layout})
            for k in T.serial(
                4,
                annotations={"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]},
            ):
                with T.block("load"):
                    T.reads(A[k, 0:8, 0:16])
                    T.writes(shared[0:8, 0:16])
                    for i in T.serial(8):
                        for j in T.serial(16):
                            shared[i, j] = A[k, i, j]
                with T.block("store"):
                    T.reads(shared[0:8, 0:16])
                    T.writes(B[k, 0:8, 0:16])
                    for i in T.serial(8):
                        for j in T.serial(16):
                            B[k, i, j] = shared[i, j]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    block = _find_block_with_layout_map(mod["main"])
    shared = next(buf for buf in block.alloc_buffers if buf.scope() == "shared.dyn")
    layout_map = block.annotations["layout_map"]

    assert [int(dim) for dim in shared.shape] == [2, 8, 16]
    assert list(layout_map[shared.data].get_input_shape()) == [2, 8, 16]
    assert layout_map[shared.data].is_equal(layout.expand([2]))


if __name__ == "__main__":
    tilelang.testing.main()
