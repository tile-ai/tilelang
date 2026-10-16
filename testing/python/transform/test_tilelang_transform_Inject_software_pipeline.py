from collections import Counter

import pytest

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
from tilelang.layout import Layout
import tilelang.testing
from tilelang.testing.ir import collect_calls, collect_nodes


def _check(original, transformed):
    func = original
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    mod = tl.transform.Simplify()(mod)
    tvm.ir.assert_structural_equal(mod["main"], transformed.with_attr("global_symbol", "main"), True)


def _count_attrs_and_calls(func):
    attr_count = Counter(node.attr_key for node in collect_nodes(func, tvm.tirx.AttrStmt))
    call_count = Counter(call.op.name for call in collect_nodes(func, tvm.tirx.Call) if isinstance(call.op, tvm.ir.Op))
    return attr_count, call_count


def _collect_attr_values(func, attr_key):
    return [int(value) for value in _collect_attr_value_nodes(func, attr_key) if isinstance(value, tvm.tirx.IntImm)]


def _collect_attr_value_nodes(func, attr_key):
    return [node.value for node in collect_nodes(func, tvm.tirx.AttrStmt) if node.attr_key == attr_key]


def _collect_call_annotation_values(root, op_name, annotation_key):
    return [call.annotations[annotation_key] for call in collect_calls(root, op=op_name) if annotation_key in call.annotations]


def _collect_wait_args(func):
    return [
        int(call.args[0])
        for call in collect_calls(func, op="tirx.ptx_wait_group")
        if len(call.args) == 1 and isinstance(call.args[0], tvm.tirx.IntImm)
    ]


def _find_pipelined_loop(func):
    loops = [loop for loop in collect_nodes(func, tvm.tirx.For) if "tl_pipelined_num_stages" in loop.annotations]
    assert loops, "Expected at least one loop annotated with tl_pipelined_num_stages"
    return loops[0]


def _count_copy_calls_with_annotation(func, annotation_key):
    calls = collect_calls(func, op="tl.tileop.copy") + collect_calls(func, op="tl.tileop.async_copy")
    values = [call.annotations.get(annotation_key) for call in calls]
    return sum(isinstance(value, tvm.tirx.IntImm) and int(value) != 0 for value in values), len(calls)


def _find_block_with_layout_map(func):
    blocks = [block for block in collect_nodes(func, tvm.tirx.SBlock) if "layout_map" in block.annotations]
    assert blocks, "Expected at least one block with layout_map"
    return blocks[0]


def _find_alloc_buffer(func, data):
    buffers = [node.buffer for node in collect_nodes(func, tvm.tirx.AllocBuffer) if node.buffer.data.same_as(data)]
    assert len(buffers) == 1, f"Expected exactly one allocation for {data}, got {len(buffers)}"
    return buffers[0]


def _assert_single_stage_pipeline(func):
    loops = collect_nodes(func, tvm.tirx.For)
    assert all("tl_pipelined_num_stages" not in loop.annotations for loop in loops)
    (pipeline_loop,) = [loop for loop in loops if loop.kind == tvm.tirx.ForKind.SERIAL]
    assert int(pipeline_loop.min) == 0 and int(pipeline_loop.extent) == 4
    (frag,) = [
        buf
        for block in collect_nodes(func, tvm.tirx.SBlock)
        for buf in block.alloc_buffers
        if buf.scope() == "local.fragment" and buf.dtype == "float16"
    ]
    assert [int(dim) for dim in frag.shape] == [4, 128]
    (load,) = [load for load in collect_nodes(pipeline_loop.body, tvm.tirx.BufferLoad) if load.buffer.data.same_as(frag.data)]
    (read_loop,) = [
        loop
        for loop in collect_nodes(pipeline_loop.body, tvm.tirx.For)
        if any(node.same_as(load) for node in collect_nodes(loop.body, tvm.tirx.BufferLoad))
    ]
    assert read_loop.kind == tvm.tirx.ForKind.PARALLEL and int(read_loop.extent) == 128
    tvm.ir.assert_structural_equal(load.indices, tvm.runtime.convert([pipeline_loop.loop_var, read_loop.loop_var]))


def _assert_no_undefined_vars(func):
    # Buffer data variables are bound by the PrimFunc buffer_map, separately
    # from the handle parameters seen by a body-only analysis.
    bound = [*func.params, *(buf.data for buf in func.buffer_map.values())]
    assert not tvm.tirx.analysis.undefined_vars(func.body, bound)


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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    mod = tl.transform.Simplify()(mod)

    attrs, calls = _count_attrs_and_calls(mod["main"])
    assert attrs.get("async_commit_queue_scope", 0) == 0
    assert attrs.get("async_wait_queue_scope", 0) == 0
    assert attrs.get("async_wait_inflight_count", 0) == 0
    assert calls.get("tirx.ptx_commit_group", 0) > 0
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.LowerOpaqueBlock()(mod)
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)
    mod = tl.transform.LowerOpaqueBlock()(mod)
    mod = tl.transform.Simplify()(mod)

    attrs, calls = _count_attrs_and_calls(mod["main"])
    assert attrs.get("async_commit_queue_scope", 0) == 0
    assert calls.get("tirx.ptx_commit_group", 0) == 2


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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    loop = _find_pipelined_loop(mod["main"])
    phases = _collect_call_annotation_values(loop, "tl.tileop.gemm", "tl.pipeline_mbar_phase_expr")

    assert len(phases) == 1
    expected = (loop.loop_var - loop.min) % 2
    assert tvm.arith.Analyzer().can_prove_equal(phases[0], expected)


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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tl.transform.Simplify()(mod)

    func = mod["main"]
    attrs, calls = _count_attrs_and_calls(func)
    assert calls.get("tirx.ptx_wait_group", 0) == 0
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    block = _find_block_with_layout_map(mod["main"])
    layout_map = block.annotations["layout_map"]
    shared_data = next(iter(layout_map.keys()))
    shared = _find_alloc_buffer(mod["main"], shared_data)

    assert [int(dim) for dim in shared.shape] == [2, 8, 16]
    assert list(layout_map[shared.data].get_input_shape()) == [2, 8, 16]
    assert layout_map[shared.data].is_equal(layout.expand([2]))


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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    func = mod["main"]
    block = _find_block_with_layout_map(func)
    layout_map = block.annotations["layout_map"]
    shared_data = next(iter(layout_map.keys()))
    if flat_alloc:
        shared = _find_alloc_buffer(func, shared_data)
    else:
        shared = next(buffer for buffer in block.alloc_buffers if buffer.data.same_as(shared_data))

    capacity = max(first, second)
    assert [int(dim) for dim in shared.shape] == [capacity, 8, 16]
    assert layout_map[shared.data].is_equal(layout.expand([capacity]))

    analyzer = tvm.arith.Analyzer()
    moduli = set()
    accesses = [
        node
        for node_type in (tvm.tirx.BufferLoad, tvm.tirx.BufferStore)
        for node in collect_nodes(func, node_type)
        if node.buffer.data.same_as(shared.data)
    ]
    for node in accesses:
        assert node.buffer.same_as(shared)
        assert len(node.indices) == 3
        version = analyzer.simplify(node.indices[0])
        bound = analyzer.const_int_bound(version)
        assert 0 <= bound.min_value <= bound.max_value < capacity
        if isinstance(version, tvm.tirx.FloorMod):
            moduli.add(int(version.b))
    for node in collect_nodes(func, tvm.tirx.SBlock):
        for region in list(node.reads) + list(node.writes):
            if region.buffer.data.same_as(shared.data):
                assert region.buffer.same_as(shared)
                assert len(region.region) == 3

    assert accesses
    assert moduli == {stages for stages in (first, second) if stages > 1}


def _collect_leading_bind_names(mod, block_names):
    leading_binds = {name: [] for name in block_names}
    for node in collect_nodes(mod["main"], tvm.tirx.SBlock):
        if str(node.name_hint) not in leading_binds:
            continue
        body = node.body
        assert isinstance(body, tvm.tirx.SeqStmt), f"{node.name_hint} body should start with scalar Bind"
        names = []
        for stmt in body.seq:
            if not isinstance(stmt, tvm.tirx.Bind):
                break
            names.append(str(stmt.var.name))
        assert names, f"{node.name_hint} body should start with scalar Bind"
        leading_binds[str(node.name_hint)].append(names)

    return leading_binds


def _collect_direct_bind_names(mod):
    return [str(node.body.var.name) for node in collect_nodes(mod["main"], tvm.tirx.SBlock) if isinstance(node.body, tvm.tirx.Bind)]


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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    with pytest.raises(tvm.TVMError, match="cannot be replayed safely"):
        tl.transform.InjectSoftwarePipeline()(mod)


def test_inject_software_pipeline_schedules_bind_that_reads_pipeline_buffer():
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
                value: T.float32 = shared[tx]
                with T.sblock("store"):
                    T.reads()
                    T.writes(B[base + tx])
                    B[base + tx] = value

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    leading_binds = _collect_leading_bind_names(mod, {"copy", "store"})
    direct_binds = _collect_direct_bind_names(mod)

    assert leading_binds["copy"]
    assert leading_binds["store"]
    assert all(names[0].startswith("base") for names in leading_binds["copy"])
    assert all(names[0].startswith("base") for names in leading_binds["store"])
    assert any(name.startswith("value") for name in direct_binds)
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
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

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)

    leading_binds = _collect_leading_bind_names(mod, {"copy", "store"})

    assert leading_binds["copy"]
    assert leading_binds["store"]
    for names in leading_binds["copy"] + leading_binds["store"]:
        assert len(names) >= 2
        assert names[0].startswith("base")
        assert names[1].startswith("offset")
    _assert_no_undefined_vars(mod["main"])


def test_inject_software_pipeline_clamps_short_symbolic_epilogue():
    """Symbolic min (blockIdx.y>>1 style) with short remaining trips.

    Mirrors GQA bwd causal ``loop_st``.  When ``by == 14`` or ``15``, the
    trip count is below the three pipeline stages.  The epilogue must therefore
    begin at ``loop_st + 2``; otherwise it replays pipeline stages that overlap
    the prologue.
    """

    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.float32),
        C: T.Tensor((16, 16), T.float32),
    ):
        for by in T.thread_binding(0, 16, thread="blockIdx.y"):
            for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
                B = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
                loop_st = by // 2
                for i in T.unroll(
                    loop_st,
                    8,
                    annotations={
                        "software_pipeline_stage": [0, 1, 2],
                        "software_pipeline_order": [0, 1, 2],
                    },
                ):
                    with T.sblock("copy"):
                        T.reads(A[tx, i])
                        T.writes(B[tx, 0])
                        B[tx, 0] = A[tx, i]
                    with T.sblock("compute"):
                        T.reads(B[tx, 0])
                        T.writes(B[tx, 0])
                        B[tx, 0] = B[tx, 0] * T.float32(2.0)
                    with T.sblock("consume"):
                        T.reads(B[tx, 0])
                        T.writes(C[tx, i])
                        C[tx, i] = B[tx, 0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    # The clamped tail starts at loop_st + max(remaining_trips, 2).  It is a
    # drain loop, so it must not retain steady-state pipeline metadata.
    clamped_tail_loops = []

    def _visit_for(node):
        if isinstance(node, tvm.tirx.For) and "max(" in str(node.extent):
            clamped_tail_loops.append(node)

    post_order_visit(mod["main"].body, _visit_for)
    assert clamped_tail_loops
    assert all(loop.kind == tvm.tirx.ForKind.SERIAL for loop in clamped_tail_loops)
    assert all("tl_pipelined_num_stages" not in loop.annotations for loop in clamped_tail_loops)
    for loop in clamped_tail_loops:
        stage_distance = tvm.tirx.IntImm(loop.min.dtype, 2)
        loop_end = tvm.tirx.IntImm(loop.min.dtype, 8)
        remaining_trips = loop_end - loop.min
        clamped_drain_start = loop.min + tvm.tirx.max(remaining_trips, stage_distance)
        pipeline_loop_end = loop.min + remaining_trips
        expected_drain_extent = pipeline_loop_end + stage_distance - clamped_drain_start
        tvm.ir.assert_structural_equal(loop.extent, expected_drain_extent)


if __name__ == "__main__":
    tilelang.testing.main()
