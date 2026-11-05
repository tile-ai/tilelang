"""CUDA memory lowering must retain native addresses, masks, and value bits."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.ir import prim
from tvm_tirx.tirx.layout import S, TileLayout


def _tensor(name, dtype="float32", *, scope="global", stride=1, offset=0):
    return ir.Var(name, tirx.TensorType((128,), dtype, scope=scope, layout=TileLayout(S[128:stride] + offset)))


def _index(lanes, base=0, stride=1):
    base = base if isinstance(base, ir.Expr) else ir.IntImm("int32", base)
    return base if lanes == 1 else prim.Ramp(base, ir.IntImm("int32", stride), lanes)


def _function(params, body, *, target="cuda"):
    attrs = {"global_symbol": "main", "calling_conv": 2}
    if target is not None:
        attrs["target"] = tvm_tirx.target.Target(target)
    return tirx.Function(params, body, ret_type=ir.PrimType("void"), attrs=ir.DictAttrs(attrs))


def _apply(function, *, ordinary=False, predicated=False):
    with tvm_tirx.transform.PassContext(config={"tl.enable_lower_ldgstg": ordinary, "tl.enable_lower_ldgstg_predicated": predicated}):
        result = tvm_ffi.get_global_func("tl.cuda.transform.LowerLDGSTG")()(tvm_tirx.IRModule({"main": function}))["main"]
    assert tirx.analysis.check_well_formed(result)
    return result


def _nodes(value, cls):
    result = []
    tvm_ffi.structural_walk(value, (cls, result.append), order="pre")
    return result


def _calls(value, name):
    return [call for call in _nodes(value, ir.Call) if isinstance(call.op, ir.Op) and call.op.name == name]


def _copy(lanes=1, *, dtype="float32", source_scope="global", source_stride=1, source_offset=0):
    source = _tensor("A", dtype, scope=source_scope, stride=source_stride, offset=source_offset)
    output = _tensor("B", dtype)
    index = _index(lanes)
    return source, output, ir.TensorStore(output, [index], ir.TensorLoad(source, [index]))


@pytest.mark.parametrize(
    "dtype,lanes,bits",
    [("float32", 1, 32), ("float32", 2, 64), ("float32", 4, 128), ("float32", 8, 256), ("float16", 8, 128), ("uint8", 16, 128)],
)
def test_ldgstg_pack_width_address_mask_and_value_type(dtype, lanes, bits):
    source, output, store = _copy(lanes, dtype=dtype, source_offset=16)
    before = _function([source, output], [store])
    result = _apply(before, ordinary=True)
    (load,) = _calls(result.body, f"tl.ldg{bits}")
    (write,) = _calls(result.body, f"tl.stg{bits}")
    assert load.ty == ir.PrimType("uint32" if bits == 32 else f"uint32x{bits // 32}")
    assert write.args[1].ty == load.ty
    for call, tensor, mask in [(load, source, 1), (write, output, 2)]:
        pointer = call.args[0]
        assert pointer.op.name == "tl.access_ptr"
        assert isinstance(pointer.args[0], ir.TensorLoad)
        assert pointer.args[0].source.same_as(tensor)
        assert int(pointer.args[1]) == lanes and int(pointer.args[2]) == mask
        assert pointer.ty.element_type == tensor.ty.dtype
    assert isinstance(before.body.seq[0], ir.TensorStore)


@pytest.mark.parametrize("target", [None, "c", {"kind": "cuda", "keys": ["cutedsl"]}])
def test_non_cuda_and_cutedsl_targets_do_not_lower(target):
    source, output, store = _copy(4)
    before = _function([source, output], [store], target=target)
    result = _apply(before, ordinary=True, predicated=True)
    assert tvm_ffi.structural_equal(before, result)


def test_lowering_is_opt_in():
    source, output, store = _copy(4)
    before = _function([source, output], [store])
    assert tvm_ffi.structural_equal(before, _apply(before))


@pytest.mark.parametrize("lanes", [1, 2, 4, 8])
@pytest.mark.parametrize("ordinary", [False, True])
def test_nested_predicated_load_keeps_both_guards_and_does_not_leak(lanes, ordinary):
    source, output, _ = _copy(lanes)
    outer, inner = ir.Var("outer", "bool"), ir.Var("inner", "bool")
    load = ir.TensorLoad(source, [_index(lanes)])
    zero = ir.FloatImm("float32", 0.0)
    if lanes > 1:
        zero = prim.Broadcast(zero, lanes)
    selected = prim.if_then_else(inner, load, zero)
    store = ir.TensorStore(output, [_index(lanes)], selected)
    before = _function([source, output, outer, inner], [ir.If(outer, [store], None), store])
    result = _apply(before, ordinary=ordinary, predicated=True)
    loads = _calls(result.body, f"tl.ldg{32 * lanes}")
    assert len(loads) == 2
    assert tvm_ffi.structural_equal(loads[0].args[-1], prim.And(outer, inner))
    assert tvm_ffi.structural_equal(loads[1].args[-1], inner)
    writes = _calls(result.body, f"tl.stg{32 * lanes}")
    assert len(writes) == (2 if ordinary else 1)
    assert writes[0].args[-1].same_as(outer)
    assert not _nodes(result.body, ir.If)


@pytest.mark.parametrize("source_scope", ["shared", "local"])
def test_predicated_store_keeps_shared_or_local_load_inside_branch(source_scope):
    source, output, store = _copy(4, source_scope=source_scope)
    guard = ir.Var("guard", "bool")
    result = _apply(_function([source, output, guard], [ir.If(guard, [store], None)]), predicated=True)
    assert len(_nodes(result.body, ir.If)) == 1
    assert not _calls(result.body, "tl.stg128")
    assert len(_nodes(result.body, ir.TensorLoad)) == 1


def test_predicated_store_keeps_unsupported_global_load_inside_branch():
    source = _tensor("bytes", "uint8")
    output = _tensor("output")
    guard = ir.Var("guard", "bool")
    value = prim.Cast("float32", ir.TensorLoad(source, [_index(1)]))
    before = _function([source, output, guard], [ir.If(guard, [ir.TensorStore(output, [_index(1)], value)], None)])
    result = _apply(before, predicated=True)
    assert len(_nodes(result.body, ir.If)) == 1
    assert not _calls(result.body, "tl.stg32")


def test_stride_two_view_is_not_reinterpreted_as_contiguous_load():
    source, output, store = _copy(4, source_stride=2)
    result = _apply(_function([source, output], [store]), ordinary=True)
    assert not _calls(result.body, "tl.ldg128")
    assert len(_calls(result.body, "tl.stg128")) == 1
    loads = [load for load in _nodes(result.body, ir.TensorLoad) if load.source.same_as(source)]
    assert len(loads) == 1 and isinstance(loads[0].indices[0], prim.Ramp)


@pytest.mark.parametrize("zero", [0.0, -0.0])
def test_predicated_load_preserves_negative_zero(zero):
    source, output, _ = _copy()
    guard = ir.Var("guard", "bool")
    value = prim.if_then_else(guard, ir.TensorLoad(source, [_index(1)]), ir.FloatImm("float32", zero))
    result = _apply(_function([source, output, guard], [ir.TensorStore(output, [_index(1)], value)]), predicated=True)
    import math

    assert len(_calls(result.body, "tl.ldg32")) == (1 if math.copysign(1.0, zero) > 0 else 0)
    assert len(_calls(result.body, "prim.if_then_else")) == (0 if math.copysign(1.0, zero) > 0 else 1)


def test_potentially_trapping_store_value_stays_guarded():
    source = _tensor("A", "int32")
    output = _tensor("B", "int32")
    divisor = ir.Var("divisor", "int32")
    value = prim.truncdiv(ir.TensorLoad(source, [_index(1)]), divisor)
    before = _function([source, output, divisor], [ir.If(divisor != 0, [ir.TensorStore(output, [_index(1)], value)], None)])
    result = _apply(before, predicated=True)
    assert len(_nodes(result.body, ir.If)) == 1
    assert not _calls(result.body, "tl.stg32")


def test_address_operands_remain_addresses_and_lowering_is_idempotent():
    source, output, store = _copy(4)
    before = _function([source, output], [store])
    once = _apply(before, ordinary=True)
    twice = _apply(once, ordinary=True)
    assert tvm_ffi.structural_equal(once, twice)
    address = tirx.address_of(ir.TensorLoad(source, [_index(1)]))
    only_address = _function([source], [ir.Evaluate(address)])
    assert tvm_ffi.structural_equal(only_address, _apply(only_address, ordinary=True))


def test_memory_intrinsic_effect_is_read_state():
    for name in ["ldg32", "ldg64", "ldg128", "ldg256", "lds32", "lds64", "lds128"]:
        assert int(ir.Op.get("tl." + name).get_attr("TCallEffectKind")) == 2


@pytest.mark.parametrize("sync,trigger", [(False, False), (False, True), (True, False), (True, True)])
def test_pdl_flags_preserve_calls_and_mark_native_function(sync, trigger):
    statements = []
    for enabled, name in [(sync, "tl.pdl_sync"), (trigger, "tl.pdl_trigger")]:
        if enabled:
            statements.append(ir.Evaluate(ir.Call(name, [], ty_args=[ir.PrimType("void")])))
    before = _function([], statements)
    result = tvm_ffi.get_global_func("tl.cuda.transform.MarkCudaSyncCalls")(True)(tvm_tirx.IRModule({"main": before}))["main"]
    assert tirx.analysis.check_well_formed(result)
    assert bool(result.attrs.get("has_cuda_pdl_sync", 0)) == sync
    assert bool(result.attrs.get("has_cuda_pdl_trigger", 0)) == trigger
    assert tvm_ffi.structural_equal(result.body, before.body)


@pytest.mark.parametrize("name", ["tl.pdl_sync", "tl.pdl_trigger"])
def test_pdl_calls_reject_unsupported_target(name):
    before = _function([], [ir.Evaluate(ir.Call(name, [], ty_args=[ir.PrimType("void")]))])
    with pytest.raises(ValueError, match="PDL is not supported"):
        tvm_ffi.get_global_func("tl.cuda.transform.MarkCudaSyncCalls")(False)(tvm_tirx.IRModule({"main": before}))


def test_pdl_sync_rejects_readonly_cache_intrinsic():
    source = _tensor("A")
    readonly = ir.Call("tl.__ldg", [tirx.address_of(ir.TensorLoad(source, [_index(1)]))], ty_args=[ir.PrimType("float32")])
    before = _function([source], [ir.Evaluate(readonly), ir.Evaluate(ir.Call("tl.pdl_sync", [], ty_args=[ir.PrimType("void")]))])
    with pytest.raises(ValueError, match="Cannot invoke __ldg function with pdl_sync"):
        tvm_ffi.get_global_func("tl.cuda.transform.MarkCudaSyncCalls")(True)(tvm_tirx.IRModule({"main": before}))
