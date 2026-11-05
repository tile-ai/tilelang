from collections import Counter

import pytest

import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tilelang.analysis._visitor import tilelang_metadata
from tilelang.language._native import dtype_of
import tilelang as tl
import tilelang.language as T
from tilelang.layout import Layout
import tilelang.testing
from tilelang.testing.ir import collect_calls, collect_nodes


def _check(original, transformed):
    func = original
    mod = tvm_tirx.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    assert tirx.analysis.check_well_formed(mod["main"])
    mod = tl.transform.Simplify()(mod)
    tvm_tirx.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"), True)


@pytest.mark.parametrize("stages,orders", [([1, 0], [0, 1]), ([0, 0], [1, 0])])
@pytest.mark.parametrize("offset", [1, 32])
@pytest.mark.parametrize("thread_form", ["unmaterialized", "materialized", "native"])
def test_inject_software_pipeline_checks_cross_thread_raw_reordering(stages, orders, offset, thread_form):
    @T.prim_func
    def before(B: T.Tensor((64,), T.int32)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((64,), T.int32)
            shared[tx + 32] = 0
            for k in T.serial(2, annotations={"software_pipeline_stage": stages, "software_pipeline_order": orders}):
                shared[tx] = k
                B[k * 32 + tx] = shared[tx + offset]

    mod = tvm_tirx.IRModule.from_expr(before)
    if thread_form != "unmaterialized":
        mod = tl.transform.MaterializeKernelLaunch(lower_thread_binding=True)(mod)
    if thread_form == "native":
        # The old AttrStmt/For thread forms both become launch_thread regions.
        # Rebuild that native region explicitly so the collector is checked on
        # a fresh body parameter and its rewritten uses as well.
        def renew_thread(node):
            if node.op.name != "tirx.launch_thread":
                return node
            previous = node.body_params[0]
            variable = ir.Var(previous.name, previous.ty, previous.loc)
            body = tvm_ffi.structural_map(node.body, (ir.Var, lambda value: variable if value.same_as(previous) else value))
            return ir.RegionStmt(node.op, node.args, [variable], node.attrs, body, node.result_vars, node.loc)

        function = mod["before"]
        mod = tvm_tirx.IRModule.from_expr(function.with_body(tvm_ffi.structural_map(function.body, (ir.RegionStmt, renew_thread))))
    # Without materialized thread bounds, even the two halves may overlap.
    if offset == 1 or thread_form == "unmaterialized":
        with pytest.raises(RuntimeError, match="later stage|cannot be reordered"):
            tl.transform.InjectSoftwarePipeline()(mod)
    else:
        tl.transform.InjectSoftwarePipeline()(mod)


def _count_attrs_and_calls(func):
    attr_count = Counter(
        str(metadata["key"]) for region in collect_nodes(func, ir.RegionStmt) if (metadata := tilelang_metadata(region)) is not None
    )
    call_count = Counter(call.op.name for call in collect_nodes(func, ir.Call) if isinstance(call.op, ir.Op))
    return attr_count, call_count


def _collect_attr_values(func, attr_key):
    return [int(value) for value in _collect_attr_value_nodes(func, attr_key) if isinstance(value, ir.IntImm)]


def _collect_attr_value_nodes(func, attr_key):
    return [
        metadata["value"]
        for node in collect_nodes(func, ir.RegionStmt)
        if (metadata := tilelang_metadata(node)) is not None and metadata.get("key") == attr_key
    ]


def _call_annotations(call):
    if call.attrs and "annotation_schema" in call.attrs:
        return tvm_ffi.get_global_func("tl.DecodeAnnotations")(call.args[-1], call.attrs["annotation_schema"])
    return {}


def _collect_call_annotation_values(root, op_name, annotation_key):
    return [value for call in collect_calls(root, op=op_name) if (value := _call_annotations(call).get(annotation_key)) is not None]


def _collect_wait_args(func):
    return [
        int(call.args[0])
        for call in collect_calls(func, op="tl.ptx_wait_group")
        if len(call.args) == 1 and isinstance(call.args[0], ir.IntImm)
    ]


def _find_pipelined_loop(func):
    loops = [loop for loop in collect_nodes(func, ir.For) if "tl_pipelined_num_stages" in loop.annotations]
    assert loops, "Expected at least one loop annotated with tl_pipelined_num_stages"
    return loops[0]


def _count_copy_calls_with_annotation(func, annotation_key):
    calls = collect_calls(func, op="tl.tileop.copy") + collect_calls(func, op="tl.tileop.async_copy")
    values = [_call_annotations(call).get(annotation_key) for call in calls]
    return sum(isinstance(value, (ir.IntImm, int)) and int(value) != 0 for value in values), len(calls)


def _find_layout_map(func):
    layouts = [
        metadata["layout_map"] for call in collect_calls(func, op="tl.annotation") if "layout_map" in (metadata := tilelang_metadata(call))
    ]
    assert len(layouts) == 1, "Expected exactly one typed layout annotation"
    return layouts[0]


def _allocations(func):
    return [
        bind.var
        for bind in collect_nodes(func, ir.Bind)
        if isinstance(bind.value, ir.Call) and isinstance(bind.value.op, ir.Op) and bind.value.op.name == "tirx.alloc_tensor"
    ]


def _find_alloc_buffer(func, tensor):
    buffers = [buffer for buffer in _allocations(func) if buffer.same_as(tensor)]
    assert len(buffers) == 1, f"Expected exactly one allocation for {tensor}"
    return buffers[0]


def _assert_single_stage_pipeline(func):
    loops = collect_nodes(func, ir.For)
    assert all("tl_pipelined_num_stages" not in loop.annotations for loop in loops)
    (pipeline_loop,) = [loop for loop in loops if loop.kind == ir.ForKind.DEFAULT]
    assert int(pipeline_loop.begin) == 0 and int(pipeline_loop.end) == 4
    (frag,) = [buf for buf in _allocations(func) if str(buf.ty.scope) == "local.fragment" and str(dtype_of(buf)) == "float16"]
    assert [int(dim) for dim in frag.ty.shape] == [4, 128]
    (load,) = [load for load in collect_nodes(pipeline_loop.body, ir.TensorLoad) if load.source.same_as(frag)]
    (read_loop,) = [
        loop
        for loop in collect_nodes(pipeline_loop.body, ir.For)
        if any(node.same_as(load) for node in collect_nodes(loop.body, ir.TensorLoad))
    ]
    assert read_loop.kind == ir.ForKind.PARALLEL and int(read_loop.end - read_loop.begin) == 128
    assert tvm_ffi.structural_equal(load.indices, tvm_ffi.convert([pipeline_loop.var, read_loop.var]))


def _assert_no_undefined_vars(func):
    assert tirx.analysis.check_well_formed(func)
    assert not tirx.analysis.undefined_vars(func.body, func.params)


def test_async_pipeline_groups_multiple_copy_producers():
    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.float32),
        B: T.Tensor((16, 16), T.float32),
        C: T.Tensor((16, 16), T.float32),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            A_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
            B_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
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
                with T.sblock("copy_a"):
                    T.reads(A[tx, i])
                    T.writes(A_shared[tx, 0])
                    A_shared[tx, 0] = A[tx, i]
                with T.sblock("copy_b"):
                    T.reads(B[tx, i])
                    T.writes(B_shared[tx, 0])
                    B_shared[tx, 0] = B[tx, i]
                with T.sblock("consume"):
                    T.reads(A_shared[tx, 0], B_shared[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = A_shared[tx, 0] + B_shared[tx, 0]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    assert tirx.analysis.check_well_formed(mod["main"])
    mod = tl.transform.Simplify()(mod)

    attrs, calls = _count_attrs_and_calls(mod["main"])
    assert attrs.get("async_commit_queue_scope", 0) == 0
    assert attrs.get("async_wait_queue_scope", 0) == 0
    assert attrs.get("async_wait_inflight_count", 0) == 0
    assert calls.get("tl.ptx_commit_group", 0) > 0
    assert 1 in _collect_wait_args(mod["main"])


def test_async_pipeline_waits_for_individual_physical_commit_group():
    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.float32),
        B: T.Tensor((16, 16), T.float32),
        D: T.Tensor((16, 16), T.float32),
        C: T.Tensor((16, 16), T.float32),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            A_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
            B_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
            D_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
            for i in T.serial(
                0,
                8,
                annotations={
                    "software_pipeline_stage": [0, 3, 0, 3, 0, 3],
                    "software_pipeline_order": [1, 0, 3, 2, 5, 4],
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_async_producers": [1, 0, 1, 0, 1, 0],
                    "software_pipeline_async_producer_groups": [0, -1, 1, -1, 2, -1],
                    "tl_pipelined_num_stages": 3,
                },
            ):
                with T.sblock("copy_a"):
                    T.reads(A[tx, i])
                    T.writes(A_shared[tx, 0])
                    A_shared[tx, 0] = A[tx, i]
                with T.sblock("consume_a"):
                    T.reads(A_shared[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = A_shared[tx, 0]
                with T.sblock("copy_b"):
                    T.reads(B[tx, i])
                    T.writes(B_shared[tx, 0])
                    B_shared[tx, 0] = B[tx, i]
                with T.sblock("consume_b"):
                    T.reads(B_shared[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = C[tx, i] + B_shared[tx, 0]
                with T.sblock("copy_d"):
                    T.reads(D[tx, i])
                    T.writes(D_shared[tx, 0])
                    D_shared[tx, 0] = D[tx, i]
                with T.sblock("consume_d"):
                    T.reads(D_shared[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = C[tx, i] + D_shared[tx, 0]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr({"global_symbol": "main", "tirx.noalias": True}))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    assert tirx.analysis.check_well_formed(mod["main"])
    mod = tl.transform.Simplify()(mod)

    loop = _find_pipelined_loop(mod["main"])
    loop_waits = _collect_wait_args(loop.body)
    all_waits = _collect_wait_args(mod["main"])

    assert loop_waits[:3] == [
        8,
        8,
        8,
    ], f"Expected fine-grained physical waits, got {loop_waits}"
    assert all_waits[-9:] == [
        8,
        7,
        6,
        5,
        4,
        3,
        2,
        1,
        0,
    ], f"Expected physical tail drain waits, got {all_waits}"


def test_async_pipeline_only_wraps_producer_statements_from_explicit_group_annotations():
    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.float32),
        B: T.Tensor((16, 16), T.float32),
        C: T.Tensor((16, 16), T.float32),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            A_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
            B_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
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
                with T.sblock("fill"):
                    T.reads()
                    T.writes(A_shared[tx, 0])
                    A_shared[tx, 0] = T.float32(0)
                with T.sblock("copy_a"):
                    T.reads(A[tx, i])
                    T.writes(A_shared[tx, 0])
                    A_shared[tx, 0] = A[tx, i]
                with T.sblock("copy_b"):
                    T.reads(B[tx, i])
                    T.writes(B_shared[tx, 0])
                    B_shared[tx, 0] = B[tx, i]
                with T.sblock("consume"):
                    T.reads(A_shared[tx, 0], B_shared[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = A_shared[tx, 0] + B_shared[tx, 0]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    assert tirx.analysis.check_well_formed(mod["main"])
    mod = tl.transform.Simplify()(mod)

    attrs, calls = _count_attrs_and_calls(mod["main"])
    assert attrs.get("async_commit_queue_scope", 0) == 0
    assert calls.get("tl.ptx_commit_group", 0) == 2


def test_async_pipeline_marks_copy_ops_for_pipeline_managed_cp_async_sync():
    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.float32),
        B: T.Tensor((16, 16), T.float32),
        C: T.Tensor((16, 16), T.float32),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            A_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
            B_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
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
                T.copy(A[tx, i : i + 1], A_shared[tx, 0:1])
                T.copy(B[tx, i : i + 1], B_shared[tx, 0:1])
                C[tx, i] = A_shared[tx, 0] + B_shared[tx, 0]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    annotated, total = _count_copy_calls_with_annotation(mod["main"], "no_implicit_async_commit_wait")
    assert total > 0
    assert annotated == total


@pytest.mark.parametrize("num_stages", [2, 3])
def test_pipeline_explicit_gemm_mbar_uses_logical_epoch(num_stages):
    @T.prim_func
    def before(
        A: T.Tensor((9, 16, 16), T.float16),
        B: T.Tensor((9, 16, 16), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((16, 16), T.float16)
            B_shared = T.alloc_shared((16, 16), T.float16)
            C_local = T.alloc_fragment((16, 16), T.float32)
            mbar = T.alloc_barrier(1)
            for i in T.serial(
                3,
                9,
                annotations={
                    "software_pipeline_stage": [0, 0, num_stages - 1],
                    "software_pipeline_order": [0, 1, 2],
                    "tl_pipelined_num_stages": T.int32(num_stages),
                },
            ):
                T.copy(A[i, 0:16, 0:16], A_shared)
                T.copy(B[i, 0:16, 0:16], B_shared)
                T.gemm(A_shared, B_shared, C_local, mbar=mbar)

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    loop = _find_pipelined_loop(mod["main"])
    phases = _collect_call_annotation_values(loop, "tl.tileop.gemm", "tl.pipeline_mbar_phase_expr")

    assert len(phases) == 1
    expected = (loop.var - loop.begin) % 2
    assert tvm_tirx.sym.Analyzer().can_prove_equal(phases[0], expected)


def test_async_pipeline_does_not_mark_non_cp_async_compatible_copy():
    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.bfloat16),
        C: T.Tensor((16, 16), T.float32),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            S = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
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
                T.copy(A[tx, i : i + 1], S[tx, 0:1])
                C[tx, i] = S[tx, 0]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    annotated, total = _count_copy_calls_with_annotation(mod["main"], "no_implicit_async_commit_wait")
    assert total > 0
    assert annotated == 0


def test_async_pipeline_relaxes_loop_wait_and_descends_trailing_drain():
    @T.prim_func
    def before(A: T.Tensor((40,), T.uint8), B: T.Tensor((40,), T.uint8)):
        S = T.alloc_buffer((4,), dtype=T.uint8, scope="shared")
        for i in T.serial(
            0,
            5,
            annotations={
                "software_pipeline_stage": [0, 3],
                "software_pipeline_order": [0, 1],
                "software_pipeline_async_stages": [0],
                "software_pipeline_async_producers": [1, 0],
                "software_pipeline_async_producer_groups": [0, -1],
            },
        ):
            with T.sblock("copy"):
                T.reads(A[i * 4 : i * 4 + 4])
                T.writes(S[0:4])
                T.copy(A[i * 4 : i * 4 + 4], S[0:4])
            with T.sblock("consume"):
                T.reads(S[0:4])
                T.writes(B[i * 4 : i * 4 + 4])
                for j in range(4):
                    B[i * 4 + j] = S[j]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)

    func = mod["main"]
    loop = _find_pipelined_loop(func)
    loop_waits = _collect_wait_args(loop.body)
    all_waits = _collect_wait_args(func)

    assert loop_waits == [3], f"Expected relaxed loop wait to keep three groups in flight, got {loop_waits}"
    assert all_waits == [3, 2, 1, 0], f"Expected trailing waits to descend through the drain suffix, got {all_waits}"


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

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)

    func = mod["main"]
    attrs, calls = _count_attrs_and_calls(func)
    assert calls.get("tl.ptx_wait_group", 0) == 0
    _assert_single_stage_pipeline(func)


def test_inject_software_pipeline_expands_annotated_layout():
    layout = Layout([8, 16], lambda i, j: i * 16 + j)

    @T.prim_func
    def before(A: T.Tensor((4, 8, 16), T.float16), B: T.Tensor((4, 8, 16), T.float16)):
        with T.sblock("root"):
            shared = T.alloc_buffer((8, 16), T.float16, scope="shared.dyn")
            T.annotate_layout({shared: layout})
            for k in T.serial(
                4,
                annotations={"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]},
            ):
                with T.sblock("load"):
                    T.reads(A[k, 0:8, 0:16])
                    T.writes(shared[0:8, 0:16])
                    for i in T.serial(8):
                        for j in T.serial(16):
                            shared[i, j] = A[k, i, j]
                with T.sblock("store"):
                    T.reads(shared[0:8, 0:16])
                    T.writes(B[k, 0:8, 0:16])
                    for i in T.serial(8):
                        for j in T.serial(16):
                            B[k, i, j] = shared[i, j]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    layout_map = _find_layout_map(mod["main"])
    shared_data = next(iter(layout_map.keys()))
    shared = _find_alloc_buffer(mod["main"], shared_data)

    assert [int(dim) for dim in shared.shape] == [2, 8, 16]
    assert list(layout_map[shared].get_input_shape()) == [2, 8, 16]
    assert layout_map[shared].is_equal(layout.expand([2]))


@pytest.mark.parametrize("first,second", [(2, 4), (4, 2), (3, 1), (1, 3), (3, 3)])
@pytest.mark.parametrize("flat_alloc", [False, True])
def test_inject_software_pipeline_sibling_buffers(first, second, flat_alloc):
    layout = Layout([8, 16], lambda i, j: i * 16 + j)
    allocate = T.alloc_buffer if flat_alloc else T.sblock_alloc_buffer

    @T.prim_func
    def before(A: T.Tensor((16, 8, 16), T.float16), B: T.Tensor((16, 8, 16), T.float16)):
        with T.sblock("root"):
            shared = allocate((8, 16), T.float16, scope="shared.dyn")
            T.annotate_layout({shared: layout})
            for k in T.serial(
                8,
                annotations={"software_pipeline_stage": [0, first - 1], "software_pipeline_order": [0, 1]},
            ):
                with T.sblock("load_first"):
                    T.reads(A[k, 0:8, 0:16])
                    T.writes(shared[0:8, 0:16])
                    for i, j in T.grid(8, 16):
                        shared[i, j] = A[k, i, j]
                with T.sblock("store_first"):
                    T.reads(shared[0:8, 0:16])
                    T.writes(B[k, 0:8, 0:16])
                    for i, j in T.grid(8, 16):
                        B[k, i, j] = shared[i, j]
            for k in T.serial(
                8,
                16,
                annotations={"software_pipeline_stage": [0, second - 1], "software_pipeline_order": [0, 1]},
            ):
                with T.sblock("load_second"):
                    T.reads(A[k, 0:8, 0:16])
                    T.writes(shared[0:8, 0:16])
                    for i, j in T.grid(8, 16):
                        shared[i, j] = A[k, i, j]
                with T.sblock("store_second"):
                    T.reads(shared[0:8, 0:16])
                    T.writes(B[k, 0:8, 0:16])
                    for i, j in T.grid(8, 16):
                        B[k, i, j] = shared[i, j]
            # Ordinary scratch-buffer uses must also address the expanded view.
            shared[0, 0] = A[0, 0, 0]
            B[0, 0, 0] = shared[0, 0]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    func = mod["main"]
    layout_map = _find_layout_map(func)
    shared_data = next(iter(layout_map.keys()))
    shared = _find_alloc_buffer(func, shared_data)

    capacity = max(first, second)
    assert [int(dim) for dim in shared.shape] == [capacity, 8, 16]
    assert layout_map[shared].is_equal(layout.expand([capacity]))

    analyzer = tvm_tirx.sym.Analyzer()
    moduli = set()
    accesses = [
        node
        for node_type in (ir.TensorLoad, ir.TensorStore)
        for node in collect_nodes(func, node_type)
        if (node.source if isinstance(node, ir.TensorLoad) else node.dest).same_as(shared)
    ]
    for node in accesses:
        assert (node.source if isinstance(node, ir.TensorLoad) else node.dest).same_as(shared)
        assert len(node.indices) == 3
        version = analyzer.simplify(node.indices[0])
        bound = analyzer.const_int_bound(version)
        assert 0 <= bound.min_value <= bound.max_value < capacity
        if isinstance(version, ir.prim.FloorMod):
            moduli.add(int(version.b))
    for region in collect_nodes(func, ir.TensorRegion):
        if region.source.same_as(shared):
            assert len(region.region) == 3

    assert accesses
    assert moduli == {stages for stages in (first, second) if stages > 1}


def _collect_leading_bind_names(mod, block_names):
    leading_binds = {name: [] for name in block_names}
    for region in collect_nodes(mod["main"], ir.RegionStmt):
        metadata = tilelang_metadata(region)
        if metadata is None or metadata.get("key") != "tl.sblock":
            continue
        name = str(metadata["value"]["name"])
        if name not in leading_binds:
            continue
        # Native regions own their body definitions. Replay binds precede
        # reads/writes metadata and accesses inside this lexical scope, rather
        # than living before an old SBlockRealize wrapper.
        names = []
        for stmt in region.body.seq:
            if isinstance(stmt, ir.Bind):
                names.append(str(stmt.var.name))
            else:
                break
        assert names, f"{name} must be dominated by replayed scalar Bind definitions"
        leading_binds[name].append(names)
    return leading_binds


def test_inject_software_pipeline_replays_scalar_bind_without_annotation_slot():
    @T.prim_func
    def before(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        shared = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                4,
                annotations={
                    "software_pipeline_stage": [0, 1],
                    "software_pipeline_order": [1, 0],
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_async_producers": [1, 0],
                    "software_pipeline_async_producer_groups": [0, -1],
                },
            ):
                base: T.int32 = i * 16
                with T.sblock("copy"):
                    T.reads(A[base + tx])
                    T.writes(shared[tx])
                    shared[tx] = A[base + tx]
                with T.sblock("store"):
                    T.reads(shared[tx])
                    T.writes(B[base + tx])
                    B[base + tx] = shared[tx]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    leading_binds = _collect_leading_bind_names(mod, {"copy", "store"})

    assert leading_binds["copy"]
    assert leading_binds["store"]
    assert all(names[0].startswith("base") for names in leading_binds["copy"])
    assert all(names[0].startswith("base") for names in leading_binds["store"])
    _assert_no_undefined_vars(mod["main"])


def test_inject_software_pipeline_replays_readonly_bufferload_bind():
    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        Ids: T.Tensor((4,), T.int32),
        B: T.Tensor((128,), T.float32),
    ):
        shared = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                4,
                annotations={
                    "software_pipeline_stage": [0, 1],
                    "software_pipeline_order": [1, 0],
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_async_producers": [1, 0],
                    "software_pipeline_async_producer_groups": [0, -1],
                    "software_pipeline_replayable_scalar_binds": [1, 0, 0],
                },
            ):
                idx: T.int32 = Ids[i] * 16
                with T.sblock("copy"):
                    T.reads(A[idx + tx])
                    T.writes(shared[tx])
                    shared[tx] = A[idx + tx]
                with T.sblock("store"):
                    T.reads(shared[tx])
                    T.writes(B[idx + tx])
                    B[idx + tx] = shared[tx]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr({"global_symbol": "main", "tirx.noalias": True}))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    leading_binds = _collect_leading_bind_names(mod, {"copy", "store"})

    assert leading_binds["copy"]
    assert leading_binds["store"]
    assert all(names[0].startswith("idx") for names in leading_binds["copy"])
    assert all(names[0].startswith("idx") for names in leading_binds["store"])
    _assert_no_undefined_vars(mod["main"])


def test_inject_software_pipeline_rejects_side_effecting_replay_mask():
    @T.prim_func
    def before(
        counter: T.Tensor((1,), T.int32),
        out: T.Tensor((8,), T.int32),
    ):
        for i in T.serial(
            0,
            4,
            annotations={
                "software_pipeline_stage": [0, 1],
                "software_pipeline_order": [0, 1],
                "software_pipeline_replayable_scalar_binds": [1, 0, 0],
            },
        ):
            pos: T.int32 = T.atomic_add(counter[0], 1, return_prev=True)
            out[i * 2] = pos
            out[i * 2 + 1] = pos

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    with pytest.raises(RuntimeError, match="cannot be replayed safely"):
        tl.transform.InjectSoftwarePipeline()(mod)


@pytest.mark.parametrize("replay_input", [False, True])
def test_inject_software_pipeline_schedules_bind_that_reads_pipeline_buffer(replay_input):
    @T.prim_func
    def before(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        shared = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                4,
                annotations={
                    "software_pipeline_stage": [0, 1, 1],
                    "software_pipeline_order": [0, 1, 2],
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_async_producers": [1, 0, 0],
                    "software_pipeline_async_producer_groups": [0, -1, -1],
                    "software_pipeline_replayable_scalar_binds": [1, 0, 0, 0],
                },
            ):
                base: T.int32 = i * 16
                with T.sblock("copy"):
                    T.reads(A[base + tx])
                    T.writes(shared[tx])
                    shared[tx] = A[base + tx]
                value: T.float32 = shared[tx] + T.cast(base, T.float32) if replay_input else shared[tx]
                with T.sblock("store"):
                    T.reads()
                    T.writes(B[base + tx])
                    B[base + tx] = value

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    leading_binds = _collect_leading_bind_names(mod, {"copy", "store"})
    value_binds = [node for node in collect_nodes(mod["main"], ir.Bind) if str(node.var.name).startswith("value")]

    assert leading_binds["copy"]
    assert leading_binds["store"]
    assert all(names[0].startswith("base") for names in leading_binds["copy"])
    assert all(names[0].startswith("base") for names in leading_binds["store"])
    assert value_binds
    _assert_no_undefined_vars(mod["main"])
    assert all(not any(name.startswith("value") for name in names) for names in leading_binds["store"])


def test_inject_software_pipeline_ignores_legacy_scalar_bind_annotation_slot():
    @T.prim_func
    def before(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        shared = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                4,
                annotations={
                    "software_pipeline_stage": [3, 0, 1],
                    "software_pipeline_order": [1, 2, 0],
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_async_producers": [0, 1, 0],
                    "software_pipeline_async_producer_groups": [-1, 0, -1],
                },
            ):
                base: T.int32 = i * 16
                with T.sblock("copy"):
                    T.reads(A[base + tx])
                    T.writes(shared[tx])
                    shared[tx] = A[base + tx]
                with T.sblock("store"):
                    T.reads(shared[tx])
                    T.writes(B[base + tx])
                    B[base + tx] = shared[tx]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    leading_binds = _collect_leading_bind_names(mod, {"copy", "store"})

    assert leading_binds["copy"]
    assert leading_binds["store"]
    assert all(names[0].startswith("base") for names in leading_binds["copy"])
    assert all(names[0].startswith("base") for names in leading_binds["store"])
    _assert_no_undefined_vars(mod["main"])


def test_inject_software_pipeline_replays_scalar_bind_dependencies():
    @T.prim_func
    def before(A: T.Tensor((128,), T.float32), B: T.Tensor((128,), T.float32)):
        shared = T.alloc_buffer((16,), dtype=T.float32, scope="shared")
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            for i in T.serial(
                0,
                4,
                annotations={
                    "software_pipeline_stage": [0, 1],
                    "software_pipeline_order": [0, 1],
                },
            ):
                base: T.int32 = i * 16
                offset: T.int32 = base + tx
                with T.sblock("copy"):
                    T.reads(A[offset])
                    T.writes(shared[tx])
                    shared[tx] = A[offset]
                with T.sblock("store"):
                    T.reads(shared[tx])
                    T.writes(B[offset])
                    B[offset] = shared[tx]

    mod = tvm_tirx.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    leading_binds = _collect_leading_bind_names(mod, {"copy", "store"})

    assert leading_binds["copy"]
    assert leading_binds["store"]
    for names in leading_binds["copy"] + leading_binds["store"]:
        assert len(names) >= 2
        assert names[0].startswith("base")
        assert names[1].startswith("offset")
    _assert_no_undefined_vars(mod["main"])


@pytest.mark.parametrize("stages,orders", [([1, 0], [0, 1]), ([0, 0], [1, 0])])
@pytest.mark.parametrize(
    "shape,strides,write_index,read_index,aliases",
    [
        ((2,), (0,), (0,), (1,), True),
        ((2, 2), (1, 1), (0, 1), (1, 0), True),
        ((2,), (2,), (0,), (1,), False),
    ],
)
def test_inject_software_pipeline_checks_physical_strided_regions(stages, orders, shape, strides, write_index, read_index, aliases):
    @T.prim_func
    def original(B: T.Tensor((2,), "int32")):
        shared = T.alloc_buffer(shape, "int32", scope="shared", strides=strides)
        shared[read_index] = 0
        for k in T.serial(2, annotations={"software_pipeline_stage": stages, "software_pipeline_order": orders}):
            shared[write_index] = k
            B[k] = shared[read_index]

    mod = tvm_tirx.IRModule.from_expr(original)
    if aliases:
        with pytest.raises(RuntimeError, match="later stage|cannot be reordered"):
            tl.transform.InjectSoftwarePipeline()(mod)
    else:
        tl.transform.InjectSoftwarePipeline()(mod)


@pytest.mark.parametrize("row_stride", [0, 16, 32])
def test_inject_software_pipeline_checks_threaded_strided_regions(row_stride):
    @T.prim_func
    def original(B: T.Tensor((64,), "int32")):
        shared = T.alloc_buffer((2, 32), "int32", scope="shared", strides=(row_stride, 1))
        for tx in T.thread_binding(32, thread="threadIdx.x"):
            for k in T.serial(2, annotations={"software_pipeline_stage": [1, 0], "software_pipeline_order": [0, 1]}):
                shared[0, tx] = k
                B[k * 32 + tx] = shared[1, tx]

    mod = tvm_tirx.IRModule.from_expr(original)
    if row_stride < 32:
        with pytest.raises(RuntimeError, match="later stage"):
            tl.transform.InjectSoftwarePipeline()(mod)
    else:
        tl.transform.InjectSoftwarePipeline()(mod)


@pytest.mark.parametrize("offset,versions", [(0, 2), (1, 2), (32, 1)])
def test_inject_software_pipeline_versions_thread_regions(offset, versions):
    @T.prim_func
    def before(A: T.Tensor((64,), T.float32), B: T.Tensor((64,), T.float32)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((64,), T.float32)
            shared[tx + 32] = 0
            for k in T.serial(2, annotations={"software_pipeline_stage": [0, 1], "software_pipeline_order": [0, 1]}):
                shared[tx] = A[k * 32 + tx]
                B[k * 32 + tx] = shared[tx + offset]

    mod = tl.transform.MaterializeKernelLaunch(lower_thread_binding=True)(tvm_tirx.IRModule.from_expr(before))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    buffers = _allocations(mod["before"])
    (shared,) = [buffer for buffer in buffers if buffer.name == "shared"]
    assert [int(dim) for dim in shared.shape] == ([2, 64] if versions == 2 else [64])


if __name__ == "__main__":
    tilelang.testing.main()
