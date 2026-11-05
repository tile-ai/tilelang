"""Packed atomic plans retain physical provenance, range, and scalar fallbacks."""

import pytest
import tvm_tirx
import tilelang as tl
from tvm_tirx import ir, tirx
from tilelang.testing.ir import collect_nodes


def integer(value):
    return ir.IntImm("int32", value)


def tensor(name, dtype="float32", scope="global", alignment=64, size=64):
    return ir.Var(name, tirx.TensorType((size,), dtype, scope=scope, layout="default", data_alignment=alignment))


def pointer(data, index, kind="access", extent=None, mask=3):
    extent = integer(1) if extent is None else extent
    if kind == "address":
        return tirx.address_of(data[index])
    if kind == "raw":
        return ir.Call("tl.ptr_access", [data.data, index, extent, integer(mask)])
    return ir.Call("tl.access_ptr", [data[index], extent, integer(mask)])


def module(source, destination, lane, ptr, lanes=4, arch="sm_90", extras=(), order=0):
    order = integer(order) if isinstance(order, int) else order
    call = ir.Call("tl.atomic_add_elem_op", [ptr, source[lane], order])
    body = ir.For(lane, 0, lanes, 1, ir.ForKind.VECTORIZED, ir.Evaluate(call))
    params = [source, destination, *extras]
    func = tirx.Function(params, body).with_attr("global_symbol", "main")
    target = tvm_tirx.target.Target({"kind": "cuda", "arch": arch})
    return ir.IRModule({"main": func.with_attrs({"target": target, "tirx.noalias": True})}), target


def plan_and_emit(mod, target):
    old = ir.save_json(mod)
    with target:
        planned = tl.transform.LegalizeVectorizedLoop()(mod)
        emitted = tl.transform.VectorizeLoop()(planned)
    assert ir.save_json(mod) == old
    assert tirx.analysis.check_well_formed(emitted["main"])
    assert not tirx.analysis.undefined_vars(emitted["main"])
    for call in collect_nodes(emitted["main"], ir.Call):
        call.validate()
    widths = [
        int((loop.end - loop.begin) // loop.step) for loop in collect_nodes(planned["main"], ir.For) if loop.kind == ir.ForKind.VECTORIZED
    ]
    atoms = [
        call for call in collect_nodes(emitted["main"], ir.Call) if isinstance(call.op, ir.Op) and call.op.name.startswith("tl.atomic_add")
    ]
    return widths, atoms, emitted


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("scope", ["global", "shared"])
@pytest.mark.parametrize("arch", ["sm_80", "sm_90"])
@pytest.mark.parametrize("kind", ["access", "address", "raw"])
def test_half_pairs_keep_their_original_target_and_storage_support(dtype, scope, arch, kind):
    source, dest = tensor("A", dtype), tensor("B", dtype, scope)
    lane = ir.Var("lane", "int32")
    mod, target = module(source, dest, lane, pointer(dest, lane, kind), lanes=2, arch=arch)
    widths, atoms, _ = plan_and_emit(mod, target)
    assert widths == [2]
    assert [atom.op.name for atom in atoms] == ["tl.atomic_addx2_elem_op"]


@pytest.mark.parametrize(
    "case",
    [
        "unknown_pointer",
        "empty_extent",
        "unknown_extent",
        "varying_extent",
        "read_mask",
        "write_mask",
        "weak_alignment",
        "misaligned_byte_offset",
        "repeated_destination",
        "strided_destination",
    ],
)
def test_unsupported_atomic_pointer_contract_stays_scalar(case):
    source = tensor("A")
    dest = tensor("B", alignment=4 if case == "weak_alignment" else 64)
    lane = ir.Var("lane", "int32")
    extras = []
    extent = integer(1)
    if case == "empty_extent":
        extent = integer(0)
    elif case == "unknown_extent":
        extent = ir.Var("unknown_extent", "int32")
        extras.append(extent)
    elif case == "varying_extent":
        extent = lane + 1
    index = integer(0) if case == "repeated_destination" else lane * 2 if case == "strided_destination" else lane
    mask = 1 if case == "read_mask" else 2 if case == "write_mask" else 3
    ptr = pointer(dest, index, "raw", extent, mask)
    if case == "unknown_pointer":
        raw = ir.Var("raw", ir.PtrType(ir.PrimType("float32")))
        extras.append(raw)
        ptr = ir.Call("tl.ptr_access", [raw, lane, integer(1), integer(3)])
    elif case == "misaligned_byte_offset":
        ptr = ir.Call("tirx.ptr_byte_offset", [dest.data, lane * 4 + 4], ty=dest.data.ty)
    mod, target = module(source, dest, lane, ptr, extras=extras)
    widths, atoms, _ = plan_and_emit(mod, target)
    assert widths == []
    assert [atom.op.name for atom in atoms] == ["tl.atomic_add_elem_op"]


def test_atomic_memory_order_must_be_a_constant():
    source, dest = tensor("A"), tensor("B")
    lane = ir.Var("lane", "int32")
    call = ir.Call("tl.atomic_add_elem_op", [pointer(dest, lane), source[lane], lane])
    with pytest.raises(TypeError, match="ir.IntImm"):
        call.validate()


@pytest.mark.parametrize("kind", ["reinterpret_element_offset", "reinterpret_byte_offset"])
def test_typed_reinterpret_keeps_known_tensor_alignment_and_byte_units(kind):
    source = tensor("A")
    storage = tensor("bytes", "uint8", size=256)
    lane = ir.Var("lane", "int32")
    typed = tirx.reinterpret(storage.data, ty=ir.PtrType(ir.PrimType("float32")))
    if kind == "reinterpret_element_offset":
        ptr = ir.Call("tl.ptr_access", [typed, lane, integer(1), integer(3)])
    else:
        ptr = ir.Call("tirx.ptr_byte_offset", [typed, lane * 4], ty=typed.ty)
    mod, target = module(source, storage, lane, ptr)
    widths, atoms, _ = plan_and_emit(mod, target)
    assert widths == [4]
    assert [atom.op.name for atom in atoms] == ["tl.atomic_addx4_elem_op"]
    assert atoms[0].args[0].ty.element_type == ir.PrimType("float32")
