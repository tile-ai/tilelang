import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang._compat import stmt_functor
from tilelang._compat.tensor import TensorView

pytestmark = pytest.mark.usefixtures("native_layouts")


def apply(name, body, parameters=(), **attributes):
    function = tirx.Function(list(parameters), body, ret_type=ir.TupleType([]))
    for key, value in {"target": Target({"kind": "cuda", "arch": "sm_100"}), **attributes}.items():
        function = function.with_attr(key, value)
    return tvm_ffi.get_global_func(f"tl.cuda.transform.{name}")()(ir.IRModule({"main": function}))["main"]


def nodes(value, node_type):
    result = []
    stmt_functor.post_order_visit(value, lambda node: result.append(node) if isinstance(node, node_type) else None)
    return result


def statement(name, *arguments, attrs=None):
    return ir.Evaluate(ir.Call(name, arguments, attrs=attrs))


def barrier_sequence(barrier=None, issue=None, arrive=None, before=()):
    if barrier is None:
        barrier = ir.Var("barrier", ir.PtrType("uint64", "shared"))
    shared = ir.Var("shared", ir.PtrType("float32", "shared"))
    source = ir.Var("source", ir.PtrType("float32", "global"))
    if issue is None:
        issue = statement("tl.tma_load", shared, source, barrier, 128, 0)
    return ir.SeqStmt(
        [
            statement("tl.mbarrier_expect_tx", barrier, 128),
            *before,
            issue,
            arrive if arrive is not None else statement("tl.mbarrier_arrive", barrier),
        ]
    )


@pytest.mark.parametrize("tensor_barrier", [False, True])
@pytest.mark.parametrize("looped", [False, True])
def test_barrier_fusion_keeps_native_pointer_or_tensor_operand(tensor_barrier, looped):
    barrier = (
        TensorView.create("barrier", [1], "uint64", scope="shared").load(0)
        if tensor_barrier
        else ir.Var("barrier", ir.PtrType("uint64", "shared"))
    )
    sequence = barrier_sequence(barrier)
    if looped:
        sequence = ir.SeqStmt(
            [sequence.seq[0], ir.For(ir.Var("issue", "int32"), 0, 2, 1, ir.ForKind.DEFAULT, sequence.seq[1]), sequence.seq[2]]
        )
    result = apply("FuseMBarrierArriveExpectTx", sequence)
    calls = nodes(result, ir.Call)
    assert [call.op.name for call in calls] == ["tl.mbarrier_arrive_expect_tx", "tl.tma_load"]
    assert calls[0].args[0].same_as(barrier)
    assert calls[1].args[2].same_as(barrier)


@pytest.mark.parametrize("tensor_barrier", [False, True])
@pytest.mark.parametrize("semantics", ["", "release.cta."])
def test_barrier_fusion_preserves_native_arrival_qualifiers(tensor_barrier, semantics):
    from tvm_tirx.backend.cuda.ptx import PTXNamespace

    barrier = (
        TensorView.create("barrier", [1], "uint64", scope="shared").load(0)
        if tensor_barrier
        else ir.Var("barrier", ir.PtrType("uint64", "shared"))
    )
    pointer = tirx.address_of(barrier) if tensor_barrier else barrier
    arrival = PTXNamespace()[f"mbarrier.arrive.{semantics}shared.b64"](pointer)
    result = apply("FuseMBarrierArriveExpectTx", barrier_sequence(barrier, arrive=ir.Evaluate(arrival)))
    calls = nodes(result, ir.Call)
    fused = [call for call in calls if call.op.name == "tirx.ptx.mbarrier_arrive_expect_tx"]
    assert len(fused) == 1
    assert not any(call.op.name in ("tl.mbarrier_expect_tx", "tirx.ptx.mbarrier_arrive_nocount") for call in calls)
    ir.assert_structural_equal(fused[0].args[0], arrival.args[0])
    assert int(fused[0].args[1]) == 128
    assert [str(argument.value) for argument in fused[0].args[4:]] == [str(argument.value) for argument in arrival.args[2:]]
    ir.assert_structural_equal(result, apply("FuseMBarrierArriveExpectTx", result.body))


def test_barrier_fusion_does_not_change_native_arrival_counts():
    from tvm_tirx.backend.cuda.ptx import PTXNamespace

    barrier = ir.Var("barrier", ir.PtrType("uint64", "shared"))
    arrival = PTXNamespace()["mbarrier.arrive.shared.b64"](barrier, 2)
    body = barrier_sequence(barrier, arrive=ir.Evaluate(arrival))
    result = apply("FuseMBarrierArriveExpectTx", body)
    assert "tl.mbarrier_expect_tx" in names(result)
    assert "tirx.ptx.mbarrier_arrive" in names(result)
    assert "tirx.ptx.mbarrier_arrive_expect_tx" not in names(result)


@pytest.mark.parametrize("boundary", ["shared_sync", "mutable_bounds", "side_effect", "different_barrier", "remote_arrive", "multicast"])
def test_barrier_fusion_respects_ordering_and_barrier_identity(boundary):
    barrier = ir.Var("barrier", ir.PtrType("uint64", "shared"))
    sequence = barrier_sequence(barrier)
    expect, issue, arrive = sequence.seq
    if boundary == "shared_sync":
        sequence = ir.SeqStmt([expect, statement("tirx.gpu_storage_sync", ir.StringImm("shared")), issue, arrive])
    elif boundary == "mutable_bounds":
        bound = TensorView.create("bound", [1], "int32").load(0)
        sequence = ir.SeqStmt([expect, ir.For(ir.Var("issue", "int32"), 0, bound, 1, ir.ForKind.DEFAULT, issue), arrive])
    elif boundary == "side_effect":
        sequence = ir.SeqStmt([expect, statement("tirx.call_extern", ir.StringImm("side_effect")), issue, arrive])
    elif boundary == "different_barrier":
        sequence = ir.SeqStmt([expect, issue, statement("tl.mbarrier_arrive", ir.Var("other", barrier.ty))])
    elif boundary == "remote_arrive":
        sequence = ir.SeqStmt([expect, issue, statement("tl.ptx_arrive_cluster_barrier", barrier, 1)])
    else:
        sequence = ir.SeqStmt([expect, statement("tl.tma_load_multicast", *issue.value.args), arrive])
    result = apply("FuseMBarrierArriveExpectTx", sequence)
    assert "tl.mbarrier_arrive_expect_tx" not in [call.op.name for call in nodes(result, ir.Call)]


def gemm():
    left = TensorView.create("left", [128, 32], "float16", scope="shared")
    right = TensorView.create("right", [32, 128], "float16", scope="shared")
    output = TensorView.create("output", [128, 128], "float32", scope="local.fragment")
    arguments = [
        left.tensor,
        right.tensor,
        output.tensor,
        ir.const(False, "bool"),
        ir.const(False, "bool"),
        128,
        128,
        32,
        0,
        ir.const(True, "bool"),
        0,
        0,
        0,
    ]
    return statement("tl.tileop.tcgen05_gemm", *arguments, attrs={"use_2cta": True})


def kernel(body, dimensions):
    return ir.RegionStmt("tl.scope", [], [], {"name_hint": "tilelang_root", "cluster_dims": dimensions}, body)


@pytest.mark.parametrize("dimensions", [[2, 1, 1], [1, 2, 1]])
def test_blackwell_two_cta_annotation_is_kernel_local(dimensions):
    result = apply("LowerBlackwell2SM", ir.SeqStmt([kernel(gemm(), dimensions), kernel(ir.Evaluate(0), [1, 1, 1])]))
    first, second = nodes(result, ir.RegionStmt)
    assert first.attrs["use_2cta"]
    assert "use_2cta" not in second.attrs
    assert "use_2cta" not in result.attrs


@pytest.mark.parametrize("dimensions", [[1, 1, 1], [2, 2, 1], [1, 1, 2], [2, 1]])
def test_blackwell_two_cta_rejects_cluster_from_another_kernel(dimensions):
    with pytest.raises(ValueError, match="same kernel"):
        apply("LowerBlackwell2SM", ir.SeqStmt([kernel(ir.Evaluate(0), [2, 1, 1]), kernel(gemm(), dimensions)]))


def test_blackwell_two_cta_supports_outlined_kernel_attributes():
    result = apply("LowerBlackwell2SM", gemm(), cluster_dims=[2, 1, 1])
    assert result.attrs["use_2cta"]


def names(value):
    return [call.op.name for call in nodes(value, ir.Call)]


@pytest.mark.parametrize("scope", ["shared", "shared.dyn", "cluster"])
def test_tcgen_fences_surround_shared_sync_and_are_idempotent(scope):
    body = ir.SeqStmt([statement("tl.tcgen05_ld"), statement("tirx.gpu_storage_sync", ir.StringImm(scope))])
    result = apply("InjectTcgen05Fence", body)
    assert names(result) == ["tl.tcgen05_ld", "tl.tcgen05_before_thread_sync", "tirx.gpu_storage_sync", "tl.tcgen05_after_thread_sync"]
    ir.assert_structural_equal(result, apply("InjectTcgen05Fence", result.body))


def test_tcgen_fences_cover_wait_use_arrive_handoffs():
    barrier = ir.Var("barrier", ir.PtrType("uint64", "shared"))
    body = ir.SeqStmt(
        [
            statement("tl.mbarrier_wait_parity", barrier, 0),
            statement("tl.tcgen05_st"),
            statement("tl.mbarrier_arrive_expect_tx", barrier, 128),
        ]
    )
    result = apply("InjectTcgen05Fence", body)
    assert names(result) == [
        "tl.mbarrier_wait_parity",
        "tl.tcgen05_after_thread_sync",
        "tl.tcgen05_st",
        "tl.tcgen05_before_thread_sync",
        "tl.mbarrier_arrive_expect_tx",
    ]


def test_tcgen_fences_recognize_native_ptx_arrivals():
    from tvm_tirx.backend.cuda.ptx import PTXNamespace

    barrier = ir.Var("barrier", ir.PtrType("uint64", "shared"))
    arrival = ir.Evaluate(PTXNamespace()["mbarrier.arrive.shared.b64"](barrier))
    body = ir.SeqStmt([statement("tl.mbarrier_wait_parity", barrier, 0), statement("tl.tcgen05_ld"), arrival])
    result = apply("InjectTcgen05Fence", body)
    assert [item.value.op.name for item in result.body.seq] == [
        "tl.mbarrier_wait_parity",
        "tl.tcgen05_after_thread_sync",
        "tl.tcgen05_ld",
        "tl.tcgen05_before_thread_sync",
        "tirx.ptx.mbarrier_arrive_nocount",
    ]
    ir.assert_structural_equal(result, apply("InjectTcgen05Fence", result.body))
    boundary = ir.SeqStmt([body.seq[0], arrival, body.seq[1]])
    result = apply("InjectTcgen05Fence", boundary)
    assert "tl.tcgen05_after_thread_sync" not in names(result)


def test_tcgen_fences_do_not_cross_wait_boundaries_or_apply_to_hopper():
    barrier = ir.Var("barrier", ir.PtrType("uint64", "shared"))
    body = ir.SeqStmt(
        [statement("tl.mbarrier_wait_parity", barrier, 0), statement("tl.mbarrier_arrive", barrier), statement("tl.tcgen05_ld")]
    )
    assert names(apply("InjectTcgen05Fence", body)) == names(body)
    sync = ir.SeqStmt([body, statement("tirx.gpu_storage_sync", ir.StringImm("shared"))])
    assert names(apply("InjectTcgen05Fence", sync, target=Target({"kind": "cuda", "arch": "sm_90"}))) == names(sync)


@pytest.mark.parametrize("scope", ["shared", "shared.dyn", "global", "local"])
def test_proxy_fences_after_generic_shared_writes(scope):
    tensor = TensorView.create("tensor", [16], "float32", scope=scope)
    result = apply("InjectFenceProxy", ir.SeqStmt([tensor.store(0, ir.const(1, "float32")), statement("tl.tma_store")]))
    assert names(result).count("tl.fence_proxy_async") == (1 if scope.startswith("shared") else 0)
    ir.assert_structural_equal(result, apply("InjectFenceProxy", result.body))


@pytest.mark.parametrize("mask", [1, 2, 3])
@pytest.mark.parametrize("lowered_pointer", [False, True])
def test_proxy_opaque_pointer_writes_honor_access_masks(mask, lowered_pointer):
    tensor = TensorView.create("tensor", [16], "float32", scope="shared")
    pointer = ir.Call("tl.access_ptr", [tensor.load(0), 16, mask])
    if lowered_pointer:
        pointer = tvm_ffi.get_global_func("tl.LowerAccessPtrCall")(pointer)
    body = ir.SeqStmt([statement("tirx.call_extern", ir.StringImm("opaque"), pointer), statement("tl.tma_store")])
    result = apply("InjectFenceProxy", body)
    assert names(result).count("tl.fence_proxy_async") == bool(mask & 2)


def test_proxy_branch_merges_keep_dirty_paths():
    tensor = TensorView.create("tensor", [1], "float32", scope="shared")
    condition = ir.Var("condition", "bool")
    body = ir.SeqStmt(
        [
            ir.If(condition, tensor.store(0, ir.const(1, "float32")), statement("tl.tma_store")),
            statement("tl.tma_store"),
        ]
    )
    result = apply("InjectFenceProxy", body)
    assert names(result).count("tl.fence_proxy_async") == 1
    assert result.body.seq[-2].value.op.name == "tl.fence_proxy_async"


@pytest.mark.parametrize("mutable_condition", [False, True])
def test_proxy_branch_fences_hoist_only_across_pure_conditions(mutable_condition):
    tensor = TensorView.create("tensor", [1], "float32", scope="shared")
    flag = TensorView.create("flag", [1], "int32")
    condition = flag.load(0) > 0 if mutable_condition else ir.Var("condition", "bool")
    parameters = [flag.tensor] if mutable_condition else [condition]
    body = ir.SeqStmt(
        [
            tensor.allocate(),
            tensor.store(0, ir.const(1, "float32")),
            ir.If(condition, statement("tl.tma_store"), statement("tl.tma_store")),
        ]
    )
    result = apply("InjectFenceProxy", body, parameters)
    assert names(result).count("tl.fence_proxy_async") == (2 if mutable_condition else 1)
    if not mutable_condition:
        assert result.body.seq[-2].value.op.name == "tl.fence_proxy_async"
    ir.assert_structural_equal(result, apply("InjectFenceProxy", result.body, parameters))


def test_proxy_native_barrier_storage_does_not_count_as_data_writes():
    from tvm_tirx.backend.cuda.ptx import PTXNamespace

    barrier = ir.Var("barrier", ir.PtrType("uint64", "shared"))
    ptx = PTXNamespace()
    body = ir.SeqStmt(
        [
            ir.Evaluate(ptx["mbarrier.init.shared.b64"](barrier, 1)),
            ir.Evaluate(ptx["mbarrier.arrive.shared.b64"](barrier)),
            ir.Evaluate(ptx["cp.async.mbarrier.arrive.shared.b64"](barrier)),
            statement("tl.tma_store"),
        ]
    )
    result = apply("InjectFenceProxy", body, [barrier])
    assert "tl.fence_proxy_async" not in names(result)


@pytest.mark.parametrize("while_loop", [False, True])
def test_proxy_async_only_loops_hoist_one_fence(while_loop):
    tensor = TensorView.create("tensor", [1], "float32", scope="shared")
    body = statement("tl.tma_store")
    loop = (
        ir.While(ir.Var("keep_running", "bool"), body)
        if while_loop
        else ir.For(ir.Var("iteration", "int32"), 0, 10, 1, ir.ForKind.DEFAULT, body)
    )
    result = apply("InjectFenceProxy", ir.SeqStmt([tensor.store(0, ir.const(1, "float32")), loop]))
    assert names(result).count("tl.fence_proxy_async") == 1
    assert result.body.seq[1].value.op.name == "tl.fence_proxy_async"
    assert "tl.fence_proxy_async" not in names(result.body.seq[2])


@pytest.mark.parametrize("early_exit", ["continue", "break"])
def test_proxy_loop_fixed_point_preserves_early_exit_state(early_exit):
    tensor = TensorView.create("tensor", [1], "float32", scope="shared")
    exit_statement = ir.Continue() if early_exit == "continue" else ir.Break()
    body = ir.SeqStmt(
        [
            statement("tl.tma_store"),
            tensor.store(0, ir.const(1, "float32")),
            ir.If(ir.Var("exit_now", "bool"), exit_statement, None),
            statement("tl.fence_proxy_async"),
        ]
    )
    loop = ir.For(ir.Var("iteration", "int32"), 0, 10, 1, ir.ForKind.DEFAULT, body)
    result = apply("InjectFenceProxy", ir.SeqStmt([loop, statement("tl.tma_store")]))
    first_loop = nodes(result, ir.For)[0]
    if early_exit == "continue":
        assert first_loop.body.seq[0].value.op.name == "tl.fence_proxy_async"
    assert result.body.seq[-2].value.op.name == "tl.fence_proxy_async"
