"""Packed FP4 storage keeps exact byte addresses, aliases and public types."""

import pytest
import tvm_ffi
import tvm_tirx
from tvm_tirx import ir, tirx
from tvm_tirx.tirx.layout import S, Axis

FP4 = "float4_e2m1fn"
PAIR = "float4_e2m1fnx2"


def integer(value, dtype="int32"):
    return ir.IntImm(dtype, value)


def tensor(name, shape=(16,), dtype=FP4, scope="global", layout=None):
    return ir.Var(name, tirx.TensorType(shape, dtype, scope=scope, layout=layout))


def allocate(value):
    return ir.Bind(value, ir.Call("tirx.alloc_tensor", [], ty_args=[value.ty]))


def declare(value, data):
    return ir.Bind(value, ir.Call("tirx.decl_tensor", [data], ty_args=[value.ty]))


def access(value, index=4, extent=8, mask=1):
    index = integer(index) if isinstance(index, int) else index
    extent = integer(extent) if isinstance(extent, int) else extent
    return ir.Call("tl.access_ptr", [ir.TensorLoad(value, [index]), extent, integer(mask)])


def objects(value, kind):
    found = []
    tvm_ffi.structural_walk(value, (kind, found.append), order="pre")
    return found


def calls(value, name):
    return [expr for expr in objects(value, ir.Call) if isinstance(expr.op, ir.Op) and expr.op.name == name]


def apply(params, body, tmp_path):
    before = tirx.Function(params, body).with_attr("global_symbol", "fp4_storage")
    assert tirx.analysis.check_well_formed(before)
    source = ir.save_json(before)
    operation = tvm_ffi.get_global_func("tl.transform.RewriteFp4ToFp4x2")()
    module = operation(tvm_tirx.IRModule({"fp4_storage": before}))
    assert tirx.analysis.check_well_formed(module)
    for expr in objects(module, ir.Call):
        expr.validate()
    ir.assert_structural_equal(operation(module), module)
    assert ir.save_json(before) == source
    (tmp_path / "before.json").write_text(source)
    (tmp_path / "after.json").write_text(ir.save_json(module["fp4_storage"]))
    return module["fp4_storage"]


def constant(value):
    return int(tvm_tirx.sym.Analyzer().simplify(value).value)


@pytest.mark.parametrize("index,extent", [(0, 2), (2, 4), (6, 8), (14, 2)])
def test_fp4_public_tensor_signature_is_preserved_and_body_addresses_are_bytes(index, extent, tmp_path):
    data = tensor("data")
    result = apply([data], [ir.Evaluate(access(data, index, extent))], tmp_path)
    assert result.params[0].same_as(data)
    assert str(result.params[0].ty.dtype) == FP4 and int(result.params[0].ty.shape[0]) == 16
    (packed,) = [bind.var for bind in objects(result, ir.Bind) if isinstance(bind.var.ty, tirx.TensorType)]
    assert str(packed.ty.dtype) == PAIR and constant(packed.ty.shape[0]) == 8
    (pointer,) = calls(result, "tl.access_ptr")
    assert pointer.args[0].source.same_as(packed)
    assert constant(pointer.args[0].indices[0]) == index // 2
    assert constant(pointer.args[1]) == extent // 2 and int(pointer.args[2]) == 1
    # Enumerate both nibble lanes of every transferred byte independently.
    original_bits = [4 * scalar for scalar in range(index, index + extent)]
    packed_bits = [8 * byte + nibble * 4 for byte in range(index // 2, (index + extent) // 2) for nibble in (0, 1)]
    assert original_bits == packed_bits


def test_fp4_allocations_and_scalar_ssa_values_are_retyped_together(tmp_path):
    source, destination = tensor("source"), tensor("destination", scope="shared.dyn")
    value = ir.Var("value", FP4)
    result = apply(
        [source], [allocate(destination), ir.Bind(value, ir.TensorLoad(source, [2])), ir.TensorStore(destination, [4], value)], tmp_path
    )
    (store,) = objects(result, ir.TensorStore)
    assert str(store.dest.ty.dtype) == PAIR and constant(store.indices[0]) == 2
    assert str(store.value.ty) == PAIR
    assert all(str(bind.var.ty) != FP4 for bind in objects(result, ir.Bind))


@pytest.mark.parametrize("lanes", [2, 4, 16])
def test_fp4_vector_ramp_retains_total_result_lanes(lanes, tmp_path):
    source, destination = tensor("source", (32,)), tensor("destination", (32,), scope="shared.dyn")
    index = ir.prim.Ramp(integer(4), integer(1), lanes)
    value = ir.TensorLoad(source, [index])
    result = apply([source], [allocate(destination), ir.TensorStore(destination, [index], value)], tmp_path)
    (store,) = objects(result, ir.TensorStore)
    assert store.value.ty.dtype.lanes == lanes
    assert str(store.dest.ty.dtype) == PAIR
    if lanes == 2:
        assert constant(store.indices[0]) == 2
    else:
        assert constant(store.indices[0].base) == 2 and constant(store.indices[0].lanes) == lanes // 2


def test_fp4_byte_offset_is_not_halved_with_element_offsets(tmp_path):
    data = ir.Var("data", ir.PtrType(ir.PrimType(FP4), "global"))
    shifted = ir.Call("tirx.ptr_byte_offset", [data, integer(5)], ty=data.ty)
    pointer = ir.Call("tl.ptr_access", [shifted, integer(6), integer(8), integer(1)])
    result = apply([data], [ir.Evaluate(pointer)], tmp_path)
    (address,) = calls(result, "tl.ptr_access")
    (byte_offset,) = calls(result, "tirx.ptr_byte_offset")
    assert constant(byte_offset.args[1]) == 5
    assert constant(address.args[1]) == 3 and constant(address.args[2]) == 4
    assert str(address.ty.element_type) == PAIR


def test_byte_backed_fp4_views_and_already_normalized_int8_pointer_stay_in_bytes(tmp_path):
    raw = ir.Var("raw", ir.PtrType(ir.PrimType("uint8"), "global"))
    view = tensor("view")
    typed = ir.Call("tirx.reinterpret", [raw], ty=ir.PtrType(ir.PrimType(FP4), "global"))
    typed = ir.Call("tirx.ptr_byte_offset", [typed, integer(5)], ty=typed.ty)
    bytes_pointer = ir.Call("tirx.reinterpret", [view.data], ty=ir.PtrType(ir.PrimType("int8"), "global"))
    address = ir.Call("tl.ptr_access", [bytes_pointer, integer(7), integer(4), integer(1)])
    genuine = tensor("genuine", scope="shared.dyn")
    result = apply([raw], [declare(view, typed), allocate(genuine), ir.Evaluate(address)], tmp_path)
    (pointer,) = calls(result, "tl.ptr_access")
    assert constant(pointer.args[1]) == 7 and constant(pointer.args[2]) == 4
    assert str(pointer.ty.element_type) == "int8"
    definitions = {bind.var.name: bind for bind in objects(result, ir.Bind)}
    assert str(definitions["view"].var.ty.dtype) == FP4
    ir.assert_structural_equal(definitions["view"].value.args[0], typed)
    assert str(definitions["genuine_fp4x2"].var.ty.dtype) == PAIR


def test_fp4_strided_layout_origin_has_exact_byte_mapping(tmp_path):
    layout = tirx.TileLayout(S[(2, 8) : (16, 1)] + 4)
    source = tensor("source", (2, 8), layout=layout)
    pointer = ir.Call("tl.access_ptr", [ir.TensorLoad(source, [1, 2]), integer(4), integer(1)])
    result = apply([source], [ir.Evaluate(pointer)], tmp_path)
    (address,) = calls(result, "tl.access_ptr")
    packed = address.args[0].source
    assert [constant(e) for e in packed.ty.shape] == [2, 4]
    assert [constant(it.stride) for it in packed.ty.layout.shard] == [8, 1]
    assert [constant(index) for index in address.args[0].indices] == [1, 1]
    assert constant(packed.ty.layout.offset[Axis.get("m")]) == 2
    for row in range(2):
        for column in range(0, 8, 2):
            assert (4 + row * 16 + column) // 2 == 2 + row * 8 + column // 2


def test_fp4_dynamic_even_shape_and_offset_remain_well_typed(tmp_path):
    n, k = ir.Var("n", "int32"), ir.Var("k", "int32")
    source = tensor("source", (2 * n,))
    result = apply([n, k, source], [ir.Evaluate(access(source, 2 * k, 2 * n))], tmp_path)
    (address,) = calls(result, "tl.access_ptr")
    assert tvm_tirx.sym.Analyzer().can_prove_equal(address.args[0].indices[0], k)
    assert tvm_tirx.sym.Analyzer().can_prove_equal(address.args[1], n)


@pytest.mark.parametrize("which", ["shape", "index", "extent", "origin"])
def test_fp4_rejects_odd_pair_contracts(which, tmp_path):
    layout = tirx.TileLayout(S[16:1] + 1) if which == "origin" else None
    source = tensor("source", (15,) if which == "shape" else (16,), layout=layout)
    with pytest.raises(ValueError, match="must be provably even"):
        apply([source], [ir.Evaluate(access(source, 3 if which == "index" else 2, 3 if which == "extent" else 4))], tmp_path)


@pytest.mark.parametrize("stride,lanes", [(1, 3), (2, 4)])
def test_fp4_rejects_non_pair_vector_ramps(stride, lanes, tmp_path):
    source = tensor("source")
    load = ir.TensorLoad(source, [ir.prim.Ramp(integer(2), integer(stride), lanes)])
    with pytest.raises(ValueError, match="even unit-stride Ramp"):
        apply([source], [ir.Evaluate(load)], tmp_path)


def test_fp4_raw_scalar_cannot_initialize_two_codes(tmp_path):
    destination = tensor("destination", scope="shared.dyn")
    scalar = ir.Var("scalar", FP4)
    with pytest.raises(TypeError, match="whole packed-pair value"):
        apply([scalar], [allocate(destination), ir.TensorStore(destination, [0], scalar)], tmp_path)


def test_fp4_cce_pointer_address_is_one_byte_per_pair(tmp_path):
    raw = ir.Var("raw", ir.PtrType(ir.PrimType(FP4), "global"))
    source = tensor("source")
    address = ir.Call("tirx.address_of", [ir.TensorLoad(source, [6])])
    result = apply([raw], [declare(source, raw), ir.Evaluate(address)], tmp_path).with_attr("calling_conv", 2)
    module = tvm_ffi.get_global_func("target.build.tilelang_ascend_without_compile")(
        tvm_tirx.IRModule({"fp4_storage": result}), tvm_tirx.target.Target("ascend")
    )
    source_code = module.inspect_source("asc")
    (tmp_path / "fp4_storage.asc").write_text(source_code)
    assert "float4_e2m1x2_t" in source_code and "[3]" in source_code


@pytest.mark.parametrize("stride", [0, 2])
def test_fp4_nonadjacent_inner_elements_are_not_packed_as_one_byte(stride, tmp_path):
    source = tensor("source", layout=tirx.TileLayout(S[16:stride]))
    with pytest.raises(ValueError, match="unit innermost stride"):
        apply([source], [ir.Evaluate(access(source, 2, 4))], tmp_path)
