"""VF legality is checked on native source IR, before scheduling or lowering."""

import pytest
import tvm_ffi
from tvm_tirx import ir, tirx

import tilelang.ascend.language as T
from tilelang.ascend.analysis import VFChecker
from tilelang.language._native import encode_annotations


def _tensor(name, dtype="float32", scope="global", size=64):
    return ir.Var(name, tirx.TensorType([ir.const(size)], dtype, scope=scope))


def _allocate(tensor):
    return ir.Bind(tensor, ir.Call("tirx.alloc_tensor", [], ty_args=[tensor.ty]))


def _attribute(key, value, body):
    schema, payload = encode_annotations({"node": None, "key": key, "value": value})
    return ir.RegionStmt("tl.attr", [payload], [], {"schema_version": 1, "annotation_schema": schema}, body)


def _vf(name, body, buffers=()):
    return _attribute("tl.sblock", {"name": name, "annotations": {}}, [*map(_allocate, buffers), body])


def _kernel(body, *, buffers=(), params=()):
    func = tirx.Function(params, [*map(_allocate, buffers), body], ir.NoneType()).with_attr("global_symbol", "main")
    return ir.IRModule({"main": func})


def _store(tensor, value, index=0):
    value = ir.const(value, tensor.ty.dtype.dtype) if isinstance(value, (int, float)) else value
    return ir.TensorStore(tensor, [ir.const(index) if isinstance(index, int) else index], value)


def _check(before, error=None):
    snapshot = ir.save_json(before)
    if error:
        with pytest.raises(ValueError, match=error):
            VFChecker()(before)
    else:
        ir.assert_structural_equal(VFChecker()(before), before)
    assert ir.save_json(before) == snapshot


@pytest.mark.parametrize(
    "vf, cast, error", [(None, False, None), (None, True, "DMA copies cannot perform type casting"), ("SIMT_VF", True, None)]
)
def test_copy_dtype_contract(vf, cast, error):
    a = _tensor("A", "float16")
    ub = _tensor("ub", "float32" if cast else "float16", "shared.dyn")
    body = ir.Evaluate(T.copy(a, ub))
    if vf:
        body = _vf(vf, body)
    _check(_kernel(body, buffers=[ub], params=[a]), error)


@pytest.mark.parametrize("vf", [None, "SIMT_VF", "SIMD_VF"])
def test_parallel_loop_requires_a_vf(vf):
    ub = _tensor("ub", scope="shared.dyn")
    i = ir.Var("i", "int32")
    body = ir.For(i, ir.const(0), ir.const(64), ir.const(1), ir.ForKind.PARALLEL, _store(ub, 1, i))
    if vf:
        body = _vf(vf, body)
    _check(_kernel(body, buffers=[ub]), None if vf else "Parallel loops outside VF blocks")


@pytest.mark.parametrize("vf", ["SIMT_VF", "SIMD_VF"])
@pytest.mark.parametrize("access", ["load", "store", "copy"])
def test_global_memory_access_depends_on_vf_kind(vf, access):
    a = _tensor("A")
    ub = _tensor("ub", scope="shared.dyn")
    if access == "copy":
        body = ir.Evaluate(T.copy(a, ub))
    elif access == "load":
        body = _store(ub, a[0])
    else:
        body = _store(a, ub[0])
    _check(
        _kernel(_vf(vf, body), buffers=[ub], params=[a]),
        "SIMD_VF blocks cannot access global memory" if vf == "SIMD_VF" else None,
    )


@pytest.mark.parametrize("vf", ["SIMT_VF", "SIMD_VF"])
@pytest.mark.parametrize("outside", [False, True])
def test_local_scalar_writes_cannot_require_pointer_capture(vf, outside):
    value = _tensor("value", "int32", "local.var", 1)
    body = _vf(vf, ir.Evaluate(T.fill(value, 1)), buffers=[] if outside else [value])
    _check(_kernel(body, buffers=[value] if outside else []), "pointer-type capture" if outside else None)


@pytest.mark.parametrize("vf, scope", [("SIMD_VF", "shared.dyn"), ("SIMT_VF", "shared.dyn"), ("SIMT_VF", "local")])
@pytest.mark.parametrize("outside", [False, True])
def test_vf_allocation_lifetime(vf, scope, outside):
    # Deliberate invalid def/use scope isolates this pass's error rather than
    # the parser's generic immutable-variable visibility check.
    temp = _tensor("temp", "int32", scope, 1)
    out = _tensor("out", "int32", size=1)
    block = _vf(vf, _store(temp, 1), buffers=[] if outside else [temp])
    before = _kernel(ir.SeqStmt([block, _store(out, temp[0])]), buffers=[temp] if outside else [], params=[out])
    _check(before, None if outside else "accessed outside its allocation scope")


@pytest.mark.parametrize("via_pointer", [False, True])
def test_local_scalar_alias_cannot_hide_pointer_capture(via_pointer):
    value = _tensor("value", "int32", "local.var", 1)
    alias = ir.Var("alias", value.ty)
    alias_value = ir.Call("tirx.decl_tensor", [value.data], ty_args=[alias.ty]) if via_pointer else value
    before = _kernel(
        _vf("SIMT_VF", ir.SeqStmt([ir.Bind(alias, alias_value), ir.Evaluate(T.fill(alias, 1))])),
        buffers=[value],
    )
    _check(before, "pointer-type capture")


def test_alias_retains_vf_allocation_lifetime():
    temp = _tensor("temp", "int32", "local", 1)
    alias = ir.Var("alias", temp.ty)
    out = _tensor("out", "int32", size=1)
    block = _vf("SIMT_VF", ir.Bind(alias, ir.Call("tirx.decl_tensor", [temp.data], ty_args=[alias.ty])), buffers=[temp])
    _check(_kernel(ir.SeqStmt([block, _store(out, alias[0])]), params=[out]), "accessed outside its allocation scope")


def test_same_named_allocations_have_independent_owners():
    outer = _tensor("value", "int32", "local.var", 1)
    inner = _tensor("value", "int32", "local.var", 1)
    _check(_kernel(_vf("SIMT_VF", _store(inner, 1), buffers=[inner]), buffers=[outer]))
    _check(_kernel(_vf("SIMT_VF", _store(outer, 1), buffers=[inner]), buffers=[outer]), "pointer-type capture")


def test_normalized_marker_and_named_block_are_one_scope():
    value = _tensor("value", "int32", "local.var", 1)
    marker = _attribute("tl.simtvf_scope", 1, _store(value, 1))
    _check(_kernel(_vf("SIMT_VF", marker, buffers=[value])))
    nested = _vf("SIMT_VF", _store(value, 1))
    _check(_kernel(_vf("SIMT_VF", nested, buffers=[value])), "pointer-type capture")


def test_direct_marker_is_a_vf_scope():
    value = _tensor("value", "int32", "local.var", 1)
    _check(_kernel(_attribute("tl.simtvf_scope", 1, [_allocate(value), _store(value, 1)])))


def test_registered_operator_effects_and_lexical_handlers_do_not_depend_on_spelling():
    # Register a real user operator using the native Fill builder, without the
    # built-in tl.tileop.* prefix. The query executes the actual native builder
    # and the registered lexical handler, rather than a Python operand table.
    name = "tl.testing.vf_fill"
    tvm_ffi.get_global_func("ir.RegisterOp")(name, "VF checker custom Fill operation")
    operation = ir.Op.get(name)
    fill = ir.Op.get("tl.tileop.fill")
    operation.set_signature(var_args="operands_and_annotations")
    operation.set_attr("TLOpBuilder", fill.get_attr("TLOpBuilder"))
    observed = []

    def lexical_handler(op, annotations):
        observed.append(int(annotations.get("testing.lexical", -1)))
        return op

    operation.set_attr("TLOpBlockAnnotationHandler", lexical_handler)

    def call(tensor):
        original = T.fill(tensor, 1)
        return ir.Evaluate(ir.Call(operation, original.args, attrs=original.attrs, ty=original.ty))

    value = _tensor("value", "int32", "local.var", 1)
    nested = _attribute("tl.sblock", {"name": "inner", "annotations": {"testing.lexical": 29}}, call(value))
    body = _attribute(
        "tl.sblock",
        {"name": "SIMT_VF", "annotations": {"testing.lexical": 17}},
        [_allocate(value), call(value), nested, call(value)],
    )
    _check(_kernel(body))
    assert observed == [17, 29, 17]
    _check(_kernel(_vf("SIMT_VF", call(value)), buffers=[value]), "pointer-type capture")


def test_source_dsl_vf_regions_are_accepted_without_rewriting():
    @T.prim_func
    def before(A: T.Tensor((64,), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((64,), "float32")
            T.copy(A, ub)
            with T.SimdVF():
                for i in T.Parallel(64):
                    ub[i] = ub[i] + 1
            with T.SimtVF(threads=64):
                local = T.alloc_var("int32", init=1)
                local = local + 1
            T.copy(ub, A)

    _check(ir.IRModule({"main": before}))
