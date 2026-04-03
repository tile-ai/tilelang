from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
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

    def _visit(node):
        if isinstance(node, tvm.tir.AttrStmt) and str(node.attr_key) == attr_key:
            value = node.value
            if isinstance(value, tvm.tir.IntImm):
                values.append(int(value.value))

    post_order_visit(func.body, _visit)
    return values


def _collect_attr_value_nodes(func, attr_key):
    values = []

    def _visit(node):
        if isinstance(node, tvm.tir.AttrStmt) and str(node.attr_key) == attr_key:
            values.append(node.value)

    post_order_visit(func.body, _visit)
    return values


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
    assert attrs.get("async_commit_queue_scope", 0) > 0
    assert attrs.get("async_wait_queue_scope", 0) > 0
    assert all(
        isinstance(value, tvm.tir.IntImm)
        for value in _collect_attr_value_nodes(mod["main"], "async_wait_inflight_count")
    )
    assert 1 in _collect_attr_values(mod["main"], "async_wait_inflight_count")
    assert calls.get("tir.ptx_commit_group", 0) == 0
    assert calls.get("tir.ptx_wait_group", 0) == 0


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

    attrs, _ = _count_attrs_and_calls(mod["main"])
    # Two producer statements are replicated into prologue / steady / epilogue.
    assert attrs.get("async_scope", 0) == 6
    assert attrs.get("async_commit_queue_scope", 0) == 3


if __name__ == "__main__":
    tilelang.testing.main()
