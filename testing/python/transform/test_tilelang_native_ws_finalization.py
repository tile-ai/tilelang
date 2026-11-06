"""WS metadata remains scoped until memory, synchronization and register consumers."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
from tilelang.language._annotation_codec import encode_annotations
from tilelang.testing.ir import collect_calls, collect_nodes


WS_REGION_KEYS = ("kWarpSpecializationScope", "kCustomWarpSpecialization")


def _regions(root):
    decode = tvm_ffi.get_global_func("tl.DecodeAnnotations")
    result = {}
    for region in collect_nodes(root, ir.RegionStmt):
        if region.op.name == "tl.attr":
            fields = decode(region.args[0], region.attrs["annotation_schema"])
            result[str(fields["key"])] = region
    return result


def _module(function):
    function = function.with_attr("global_symbol", "main")
    function = function.with_attr("target", Target({"kind": "cuda", "arch": "sm_90"}))
    return ir.IRModule({"main": function})


def _attribute_kernel(key):
    @T.prim_func
    def kernel(A: T.Tensor((256,), T.int32)):
        T.attr(0, key, 1)
        A[0] = 1

    return kernel


def _role_kernel(custom=False):
    @T.prim_func
    def kernel(A: T.Tensor((256,), T.int32)):
        tx = T.launch_thread("threadIdx.x", 256)
        if custom:
            T.attr(0, "kCustomWarpSpecialization", 1)
        T.attr([128, 128], "kWarpSpecializationScope", tx)
        if tx < 128:
            value = A[tx]
            A[tx] = value + 1
        else:
            value = A[tx]
            A[tx] = value + 2

    return kernel


@pytest.mark.parametrize("key", WS_REGION_KEYS)
def test_terminal_check_rejects_unconsumed_ws_regions(key):
    module = tilelang.transform.LowerSourceAnnotations()(_module(_attribute_kernel(key)))
    assert key in _regions(module)
    with pytest.raises(ValueError, match=key):
        tilelang.transform.VerifyLoweredAnnotations()(module)


@pytest.mark.parametrize("custom", [False, True])
def test_source_lowering_keeps_ws_operands_binders_and_last_consumers(custom):
    original = _module(_role_kernel(custom))
    before = _regions(original)
    early = tilelang.transform.LowerSourceAnnotations()(original)
    pending = _regions(early)
    assert set(pending) == set(before)
    for key, region in pending.items():
        previous = before[key]
        assert region.op.same_as(previous.op)
        for field in ("args", "body_params", "attrs", "result_vars", "loc"):
            assert tvm_ffi.structural_equal(getattr(region, field), getattr(previous, field))
    before_binders = collect_nodes(original, ir.Bind)
    after_binders = collect_nodes(early, ir.Bind)
    assert len(after_binders) == len(before_binders)
    assert all(before.var.same_as(after.var) for before, after in zip(before_binders, after_binders))
    assert tirx.analysis.check_well_formed(early)
    assert tirx.analysis.check_well_formed(tirx.renew_def(early["main"]))

    # The real memory and synchronization consumers still need both scopes.
    current = tilelang.transform.MergeSharedMemoryAllocations()(early)
    current = tilelang.transform.ThreadSync("shared")(current)
    current = tilelang.transform.ThreadSync("shared.dyn")(current)
    assert set(_regions(current)) == set(pending)
    with pytest.raises(ValueError, match="kWarpSpecializationScope|kCustomWarpSpecialization"):
        tilelang.transform.VerifyLoweredAnnotations()(current)
    lowered = tilelang.cuda.transform.AnnotateWarpGroupRegAlloc()(current)
    finished = tilelang.transform.VerifyLoweredAnnotations()(lowered)
    assert tvm_ffi.structural_equal(finished, lowered)
    assert not _regions(finished)
    assert tirx.analysis.check_well_formed(finished)
    assert tirx.analysis.check_well_formed(tirx.renew_def(finished["main"]))
    assert len(collect_nodes(finished, ir.TensorStore)) == 2
    assert len(collect_calls(finished, op="tl.set_max_nreg")) == (0 if custom else 2)


@pytest.mark.parametrize("key", ["tl.unconsumed_region", "barrier_init", "safe_value_map"])
def test_source_lowering_never_hides_other_pending_contracts(key):
    module = tilelang.transform.LowerSourceAnnotations()(_module(_attribute_kernel(key)))
    assert key in _regions(module)
    with pytest.raises(ValueError, match=key):
        tilelang.transform.VerifyLoweredAnnotations()(module)


def test_terminal_check_accepts_mixed_host_device_targets():
    device = _role_kernel().with_attr("global_symbol", "device")
    device = device.with_attr("target", Target({"kind": "cuda", "arch": "sm_90"}))
    host = tirx.Function([], [], ir.TupleType([]), ir.DictAttrs({"target": Target("llvm")}))
    module = ir.IRModule({"main": device, "host": host})
    early = tilelang.transform.LowerSourceAnnotations()(module)
    lowered = tilelang.cuda.transform.AnnotateWarpGroupRegAlloc()(early)
    final = tilelang.transform.VerifyLoweredAnnotations()(lowered)
    assert tvm_ffi.structural_equal(final, lowered)
    assert not _regions(final)
    assert final["host"].attrs["target"].kind.name == "llvm"
    assert final["main"].attrs["target"].kind.name == "cuda"
    assert tirx.analysis.check_well_formed(final)


@pytest.mark.parametrize("custom", [False, True])
def test_disabled_register_inference_consumes_hints_but_keeps_raw_calls(custom):
    @T.prim_func
    def kernel(A: T.Tensor((256,), T.int32)):
        tx = T.launch_thread("threadIdx.x", 256)
        T.annotate_producer_reg_dealloc(48)
        T.annotate_consumer_reg_alloc(216)
        T.disable_warp_group_reg_alloc()
        if custom:
            T.attr(0, "kCustomWarpSpecialization", 1)
        T.attr([128, 128], "kWarpSpecializationScope", tx)
        if tx < 128:
            value = A[tx]
            T.set_max_nreg(80, 0)
            A[tx] = value + 1
        else:
            value = A[tx]
            T.set_max_nreg(232, 1)
            A[tx] = value + 2

    source = tilelang.transform.LowerSourceAnnotations()(_module(kernel))
    original_calls = collect_calls(source, op="tl.set_max_nreg")
    result = tilelang.cuda.transform.AnnotateWarpGroupRegAlloc(enable_register_inference=False)(source)
    assert not _regions(result)
    calls = collect_calls(result, op="tl.set_max_nreg")
    assert [tuple(int(arg) for arg in call.args) for call in calls] == [(80, 0), (232, 1)]
    assert all(tvm_ffi.structural_equal(before, after) for before, after in zip(original_calls, calls))
    for op in ("tl.annotate_producer_reg_dealloc", "tl.annotate_consumer_reg_alloc", "tl.no_set_max_nreg"):
        assert not collect_calls(result, op=op)
    assert len([bind for bind in collect_nodes(result, ir.Bind) if isinstance(bind.value, ir.TensorLoad)]) == 2
    assert tirx.analysis.check_well_formed(result)
    assert tirx.analysis.check_well_formed(tirx.renew_def(result["main"]))
    assert tvm_ffi.structural_equal(result, tilelang.transform.VerifyLoweredAnnotations()(result))


@pytest.mark.parametrize("enable_register_inference", [False, True])
@pytest.mark.parametrize("custom", [False, True])
def test_register_owner_keeps_private_bind_and_allocation_scopes(enable_register_inference, custom):
    private = ir.Var("ws_private", "int32")
    scratch = ir.Var("ws_scratch", tirx.TensorType((1,), "int32", scope="local", layout="default"))
    body = [
        ir.Bind(private, ir.IntImm("int32", 7)),
        ir.Bind(scratch, ir.Call("tirx.alloc_tensor", [], ty_args=[scratch.ty])),
        ir.TensorStore(scratch, [0], private),
    ]

    def attribute(node, key, value, statements):
        schema, payload = encode_annotations({"node": node, "key": key, "value": value})
        return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, statements)

    scope = attribute([128, 128], "kWarpSpecializationScope", 0, body)
    if custom:
        scope = attribute(None, "kCustomWarpSpecialization", 1, [scope])
    original = _module(tirx.Function([], [scope], ir.PrimType("void")))
    assert tirx.analysis.check_well_formed(original)
    lowered = tilelang.transform.LowerSourceAnnotations()(original)
    result = tilelang.cuda.transform.AnnotateWarpGroupRegAlloc(enable_register_inference=enable_register_inference)(lowered)
    final = tilelang.transform.VerifyLoweredAnnotations()(result)["main"]
    assert not _regions(final)
    assert final.body.seq[0].op.name == "tl.lexical_scope"
    assert tirx.analysis.check_well_formed(final)
    assert tirx.analysis.check_well_formed(tirx.renew_def(final))
    assert len(collect_calls(final, op="tirx.alloc_tensor")) == 1
    assert not collect_calls(final, op="tl.set_max_nreg")
    for local in (private, scratch):
        escaped = final.with_body([*final.body.seq, ir.Evaluate(local)])
        assert not tirx.analysis.check_well_formed(escaped, assert_mode=False)


@pytest.mark.parametrize("enable_register_inference", [False, True])
def test_register_owner_does_not_complete_an_unprocessed_pipeline(enable_register_inference):
    @T.prim_func
    def kernel(A: T.Tensor((4, 8), T.float32), B: T.Tensor((4, 8), T.float32)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((8,), T.float32)
            for i in T.Pipelined(4, num_stages=2):
                T.copy(A[i, 0], shared)
                T.copy(shared, B[i, 0])

    original = _module(kernel)
    result = tilelang.cuda.transform.AnnotateWarpGroupRegAlloc(enable_register_inference=enable_register_inference)(original)
    assert tvm_ffi.structural_equal(result, original)
    assert len(collect_calls(result, op="tl.tileop.copy")) == 2
    assert not collect_calls(result, op="tl.set_max_nreg")
    assert any(int(loop.annotations.get("num_stages", 0)) == 2 for loop in collect_nodes(result, ir.For))
