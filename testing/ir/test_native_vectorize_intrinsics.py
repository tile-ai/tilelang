"""Native intrinsic vectorization preserves transactions and scalar fallback."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx

from test_native_reducer_contracts import integer, objects, tensor


def pointer(value, index, extent=1, mask=1, kind="access"):
    load = ir.TensorLoad(value, [index])
    if kind == "address":
        return ir.Call("tirx.address_of", [load])
    if kind == "raw":
        return ir.Call("tl.ptr_access", [ir.Call("tirx.tensor_data_ptr", [value]), index, integer(extent), integer(mask)])
    return ir.Call("tl.access_ptr", [load, integer(extent), integer(mask)])


def vector_function(params, lane, lanes, call, arch="sm_90"):
    loop = ir.For(lane, integer(0), integer(lanes), integer(1), ir.ForKind.VECTORIZED, [ir.Evaluate(call)])
    # These fixture tensors represent independent allocations, as supplied by
    # the production TileLang driver. Raw Function constructors do not add it.
    return tirx.Function(params, [loop], ir.NoneType()).with_attrs(
        {"target": tvm_tirx.target.Target({"kind": "cuda", "arch": arch}), "tirx.noalias": True}
    )


def vectorize(value, enabled=True, namespace="tl"):
    before = ir.save_json(value)
    result = tvm_ffi.get_global_func(namespace + ".transform.VectorizeLoop")(enabled)(ir.IRModule({"main": value}))["main"]
    assert before == ir.save_json(value)
    assert tirx.analysis.check_well_formed(result)
    assert not tirx.analysis.undefined_vars(result)
    for call in objects(result, ir.Call):
        call.validate()
    return result


def async_calls(value):
    return [call for call in objects(value, ir.Call) if isinstance(call.op, ir.Op) and call.op.name.startswith("tl.ptx_cp_async")]


@pytest.mark.parametrize("dtype,lanes", [("float16", 2), ("float16", 4), ("float16", 8), ("float32", 2), ("float32", 4), ("uint8", 4)])
@pytest.mark.parametrize("kind", ["access", "address", "raw"])
def test_cp_async_combines_exact_contiguous_elements(dtype, lanes, kind):
    source = tensor("A", [64], dtype, "global")
    destination = tensor("S", [64], dtype, "shared")
    lane = ir.Var("lane", "int32")
    call = ir.Call("tl.ptx_cp_async", [pointer(destination, lane, mask=2, kind=kind), pointer(source, lane, kind=kind), integer(1)])
    result = vectorize(vector_function([source, destination], lane, lanes, call))
    assert not objects(result, ir.For)
    (combined,) = async_calls(result)
    assert int(combined.args[2].value) == lanes
    assert isinstance(combined.args[0].ty, ir.PtrType)
    if kind != "address":
        extent_index = 1 if kind == "access" else 2
        assert int(combined.args[0].args[extent_index].value) == lanes
        assert int(combined.args[1].args[extent_index].value) == lanes


@pytest.mark.parametrize(
    "case", ["strided", "repeated", "reverse", "offset", "weak_alignment", "too_wide", "varying_predicate", "varying_metadata"]
)
def test_cp_async_unsupported_transaction_retains_scalar_loop(case):
    source = tensor("A", [64], "float32", "global")
    destination = tensor("S", [64], "float32", "shared")
    if case == "weak_alignment":
        source = ir.Var("A", tirx.TensorType([64], "float32", scope="global", data_alignment=4))
    lane = ir.Var("lane", "int32")
    index = (
        2 * lane
        if case == "strided"
        else integer(0)
        if case == "repeated"
        else (integer(3) - lane if case == "reverse" else lane + 1 if case == "offset" else lane)
    )
    args = [pointer(destination, lane, mask=2), pointer(source, index), integer(1)]
    if case == "varying_predicate":
        args.append(lane < 1)
    attrs = None
    if case == "varying_metadata":
        schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"per_lane": lane})
        attrs = {"schema_version": 1, "operand_count": len(args), "annotation_schema": schema}
        args.append(payload)
    call = ir.Call("tl.ptx_cp_async", args, attrs=attrs)
    result = vectorize(vector_function([source, destination], lane, 8 if case == "too_wide" else 2, call))
    loops = objects(result, ir.For)
    assert len(loops) == 1 and loops[0].kind == ir.ForKind.DEFAULT
    assert int(async_calls(result)[0].args[2].value) == 1


@pytest.mark.parametrize("namespace", ["tl", "tirx"])
def test_cp_async_bytes_units_and_uniform_metadata_survive(namespace):
    source = tensor("A", [64], "float16", "global")
    destination = tensor("S", [64], "float16", "shared")
    lane = ir.Var("lane", "int32")
    predicate = ir.Var("enabled", "bool")
    schema, payload = tvm_ffi.get_global_func("tl.EncodeAnnotations")({"tag": 17})
    call = ir.Call(
        "tl.ptx_cp_async_bytes",
        [pointer(destination, lane * 2, 2, 2), pointer(source, lane * 2, 2), integer(4), predicate, payload],
        ty_args=[ir.PrimType("void")],
        attrs={"schema_version": 1, "operand_count": 4, "annotation_schema": schema},
    )
    result = vectorize(vector_function([source, destination, predicate], lane, 2, call), namespace=namespace)
    (combined,) = async_calls(result)
    # The upstream closed-set vectorizer does not consume TileLang call rules.
    assert len(objects(result, ir.For)) == (0 if namespace == "tl" else 1)
    assert int(combined.args[2].value) == (8 if namespace == "tl" else 4)
    assert combined.args[3].same_as(predicate)
    ir.assert_structural_equal(call.attrs, combined.attrs)
    assert tvm_ffi.get_global_func("tl.DecodeAnnotations")(combined.args[-1], schema)["tag"] == 17


@pytest.mark.parametrize("lanes", [2, 4])
@pytest.mark.parametrize("order", [0, 3, 5])
def test_atomic_pointer_pair_preserves_memory_order(lanes, order):
    source, destination = tensor("A", [64], scope="global"), tensor("B", [64], scope="global")
    lane = ir.Var("lane", "int32")
    call = ir.Call("tl.atomic_add_elem_op", [pointer(destination, lane, mask=3), ir.TensorLoad(source, [lane]), integer(order)])
    result = vectorize(vector_function([source, destination], lane, lanes, call))
    assert not objects(result, ir.For)
    atoms = [call for call in objects(result, ir.Call) if "atomic_addx" in call.op.name]
    assert len(atoms) == 1 and atoms[0].op.name == f"tl.atomic_addx{lanes}_elem_op"
    assert all(isinstance(arg.ty, ir.PtrType) for arg in atoms[0].args[:2])
    assert int(atoms[0].args[2].value) == order


@pytest.mark.parametrize("case", ["broadcast", "computed", "same_destination", "sm80", "disabled"])
def test_atomic_scalar_fallback_keeps_all_contributions(case):
    source, destination = tensor("A", [64], scope="global"), tensor("B", [64], scope="global")
    lane = ir.Var("lane", "int32")
    value = ir.TensorLoad(source, [lane])
    if case == "broadcast":
        value = ir.FloatImm("float32", 1)
    if case == "computed":
        value = value + ir.FloatImm("float32", 1)
    dst_index = integer(0) if case == "same_destination" else lane
    call = ir.Call("tl.atomic_add_elem_op", [pointer(destination, dst_index, mask=3), value])
    result = vectorize(
        vector_function([source, destination], lane, 4, call, arch="sm_80" if case == "sm80" else "sm_90"), enabled=case != "disabled"
    )
    assert len(objects(result, ir.For)) == 1
    assert not any("atomic_addx" in call.op.name for call in objects(result, ir.Call))


@pytest.mark.parametrize("noalias", [False, True])
def test_atomic_packing_requires_independent_parameter_storage(noalias):
    source, destination = tensor("A", [64], scope="global"), tensor("B", [64], scope="global")
    lane = ir.Var("lane", "int32")
    call = ir.Call("tl.atomic_add_elem_op", [pointer(destination, lane, mask=3), source[lane]])
    function = vector_function([source, destination], lane, 4, call).with_attr("tirx.noalias", noalias)
    result = vectorize(function)
    # Without noalias, B may point one element ahead of A: batching four
    # contribution loads would discard the scalar loop's carry dependency.
    assert bool(objects(result, ir.For)) == (not noalias)
    assert any("atomic_addx4" in call.op.name for call in objects(result, ir.Call)) == noalias


@pytest.mark.parametrize("name,arity", [("prim.exp", 1), ("prim.pow", 2), ("prim.fma", 3)])
def test_declared_elementwise_math_widens_each_value_operand(name, arity):
    source, destination = tensor("A", [64], scope="global"), tensor("B", [64], scope="global")
    lane = ir.Var("lane", "int32")
    args = [source[lane], *[ir.FloatImm("float32", 2) for _ in range(arity - 1)]]
    value = ir.Call(name, args)
    body = ir.For(lane, 0, 4, 1, ir.ForKind.VECTORIZED, ir.TensorStore(destination, [lane], value))
    function = tirx.Function([source, destination], body).with_attrs(
        {"target": tvm_tirx.target.Target({"kind": "cuda", "arch": "sm_90"}), "tirx.noalias": True}
    )
    result = vectorize(function)
    assert not objects(result, ir.For)
    (call,) = [call for call in objects(result, ir.Call) if call.op.name == name]
    assert call.ty == ir.PrimType("float32x4")
    assert all(arg.ty == ir.PrimType("float32x4") for arg in call.args)
