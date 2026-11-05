"""WS metadata remains scoped until memory, synchronization and register consumers."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

import tilelang
import tilelang.language as T
from tilelang.testing.ir import collect_calls, collect_nodes


PENDING_WS = ("kWarpSpecializationScope", "kCustomWarpSpecialization")


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


@pytest.mark.parametrize("key", PENDING_WS)
def test_default_finalization_rejects_unconsumed_ws_regions(key):
    module = _module(_attribute_kernel(key))
    with pytest.raises(ValueError, match="Unconsumed TileLang region"):
        tilelang.transform.FinalizeAnnotations()(module)
    with pytest.raises(ValueError, match="Unconsumed TileLang region"):
        tilelang.transform.LowerOpaqueBlock()(module)


@pytest.mark.parametrize("custom", [False, True])
def test_early_finalization_keeps_ws_operands_binders_and_last_consumers(custom):
    original = _module(_role_kernel(custom))
    before = _regions(original)
    early = tilelang.transform.FinalizeAnnotations("cuda", allowed_pending_regions=PENDING_WS)(original)
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

    # These real consumers must still see both role scopes. Finalization is
    # deliberately after them, even when a tiny kernel needs no shared arena.
    current = tilelang.transform.MergeSharedMemoryAllocations()(early)
    current = tilelang.transform.ThreadSync("shared")(current)
    current = tilelang.transform.ThreadSync("shared.dyn")(current)
    assert set(_regions(current)) == set(pending)
    with pytest.raises(ValueError, match="Unconsumed TileLang region"):
        tilelang.transform.FinalizeAnnotations()(current)
    lowered = tilelang.cuda.transform.AnnotateWarpGroupRegAlloc()(current)
    finished = tilelang.transform.FinalizeAnnotations()(lowered)
    assert not _regions(finished)
    assert tirx.analysis.check_well_formed(finished)
    assert tirx.analysis.check_well_formed(tirx.renew_def(finished["main"]))
    assert len(collect_nodes(finished, ir.TensorStore)) == 2
    assert len(collect_calls(finished, op="tl.set_max_nreg")) == (0 if custom else 2)


@pytest.mark.parametrize("key", ["unknown_region", "barrier_init", "safe_value_map", "tl.loop.annotations"])
def test_only_known_ws_regions_can_be_deferred(key):
    with pytest.raises(ValueError, match="cannot defer region"):
        tilelang.transform.FinalizeAnnotations(allowed_pending_regions=[key])
    with pytest.raises(ValueError, match="cannot defer region"):
        tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")("", [key])


@pytest.mark.parametrize("key", ["tl.unconsumed_region", "barrier_init", "safe_value_map"])
def test_deferring_ws_never_suppresses_other_pending_contracts(key):
    module = _module(_attribute_kernel(key))
    with pytest.raises(ValueError, match="Unconsumed|must be consumed"):
        tilelang.transform.FinalizeAnnotations(allowed_pending_regions=PENDING_WS)(module)


def test_allowing_one_ws_region_does_not_allow_the_other():
    module = _module(_role_kernel(custom=True))
    with pytest.raises(ValueError, match="Unconsumed TileLang region"):
        tilelang.transform.FinalizeAnnotations(allowed_pending_regions=[PENDING_WS[0]])(module)


def test_one_argument_ffi_remains_strict_and_target_checks_remain_active():
    module = _module(_attribute_kernel(PENDING_WS[0]))
    factory = tvm_ffi.get_global_func("tl.transform.FinalizeAnnotations")
    with pytest.raises(ValueError, match="Unconsumed TileLang region"):
        factory("cuda")(module)
    with pytest.raises(ValueError, match="target disagrees"):
        tilelang.transform.FinalizeAnnotations("llvm", allowed_pending_regions=PENDING_WS)(module)
    for arguments in ((), ("", [], [])):
        with pytest.raises(TypeError, match="expects target_kind"):
            factory(*arguments)


def test_late_strict_finalization_accepts_mixed_host_device_targets():
    device = _role_kernel().with_attr("global_symbol", "device")
    device = device.with_attr("target", Target({"kind": "cuda", "arch": "sm_90"}))
    host = tirx.Function([], [], ir.TupleType([]), ir.DictAttrs({"target": Target("llvm")}))
    module = ir.IRModule({"main": device, "host": host})
    early = tilelang.transform.FinalizeAnnotations(allowed_pending_regions=PENDING_WS)(module)
    lowered = tilelang.cuda.transform.AnnotateWarpGroupRegAlloc()(early)
    final = tilelang.transform.FinalizeAnnotations()(lowered)
    assert not _regions(final)
    assert final["host"].attrs["target"].kind.name == "llvm"
    assert final["main"].attrs["target"].kind.name == "cuda"
    assert tirx.analysis.check_well_formed(final)
