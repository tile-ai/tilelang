"""Physical storage oracle for Ascend version padding through declared views."""

import itertools

import pytest
import tvm_ffi
from tvm_tirx import ir, sym, tirx
from tvm_tirx.tirx.layout import S, TileLayout

from testing.ascend._ir import scheduled_fields, substitute
from testing.ascend.transform.test_native_ascend_allocations import (
    address,
    allocate,
    annotation,
    apply,
    constant,
    declare,
    definitions,
    integer,
    objects,
    pointer,
    tensor,
    unit,
)


def offset_pointer(base, byte_offset, explicit_type=True):
    return ir.Call(
        "tirx.ptr_byte_offset",
        [base, integer(byte_offset)],
        ty=base.ty if explicit_type else None,
    )


def original_physical_bit(bit):
    slot, within = divmod(bit, 96)
    return 256 * slot + within


def assert_addresses(result, expected_bits):
    binds = definitions(result)
    # A scheduled Bind retains its original value as operand zero. Reading
    # that value does not change the owner/byte-offset oracle below.
    for var, value in list(binds.items()):
        if isinstance(value, ir.Call) and value.op.name == "tl.ascend.scheduled_bind":
            fields = scheduled_fields(ir.Bind(var, value))
            assert int(fields["stage"]) == 0 and int(fields["task"]["core_mask"]) == 5
            binds[var] = value.args[0]
    loads = objects(result, ir.TensorLoad)
    actual = []
    owners = []
    for load in loads:
        owner, offset = address(load.source, binds)
        bits = int(load.ty.dtype.bits) * int(load.ty.dtype.lanes)
        actual.append(offset * 8 + constant(load.indices[0]) * bits)
        owners.append(owner)
    assert actual == expected_bits
    assert owners and all(owner.same_as(owners[0]) for owner in owners)
    assert owners[0].name == "storage"


@pytest.mark.parametrize("scheduled", [False, True])
@pytest.mark.parametrize(
    "dtype,bits,offset,length",
    [("int8", 8, 3, 20), ("int16", 16, 4, 10), ("int32", 32, 4, 5), ("float4_e2m1fn", 4, 1, 42)],
)
def test_declared_alias_preserves_bit_addresses(scheduled, dtype, bits, offset, length):
    storage = tensor("storage", (2, 3), "int32")
    view = tensor("view", (length,), dtype)
    base = ir.Call("tirx.reinterpret", [storage.data], ty=view.data.ty)
    binds = [allocate(storage), declare(view, offset_pointer(base, offset))]
    if scheduled:
        binds = [unit(binding) for binding in binds]
    result = apply(
        "RewriteAscendBufferVersionLayout",
        [*binds, annotation({"tl.buffer_version": {storage: 2}}), *[ir.Evaluate(ir.TensorLoad(view, [i])) for i in range(length)]],
    )
    assert_addresses(result, [original_physical_bit(offset * 8 + i * bits) for i in range(length)])


@pytest.mark.parametrize("chain", ["pointer-var", "nested-offset"])
def test_declared_alias_chain_keeps_layout_offset_and_storage(chain):
    storage = tensor("storage", (2, 3), "int32")
    middle = tensor("middle", (22,), "int8")
    view = tensor("view", (2, 3), "int16", layout=TileLayout(S[(2, 3) : (5, 1)] + 1))
    pointer_var = ir.Var("pointer_alias", middle.data.ty)
    middle_base = ir.Call("tirx.reinterpret", [storage.data], ty=middle.data.ty)
    view_base = ir.Call("tirx.reinterpret", [middle.data], ty=view.data.ty)
    middle_pointer = offset_pointer(middle_base, 2)
    if chain == "nested-offset":
        middle_pointer = offset_pointer(offset_pointer(middle_base, 1), 1)
    indices = list(itertools.product(range(2), range(3)))
    result = apply(
        "RewriteAscendBufferVersionLayout",
        [
            allocate(storage),
            ir.Bind(pointer_var, middle_pointer),
            declare(middle, pointer_var),
            declare(view, view_base),
            annotation({"tl.buffer_version": {storage: 2}}),
            *[ir.Evaluate(ir.TensorLoad(view, list(i))) for i in indices],
        ],
    )
    assert_addresses(result, [original_physical_bit(16 + (1 + i * 5 + j) * 16) for i, j in indices])


@pytest.mark.parametrize("access", ["tensor", "pointer"])
def test_alias_extent_cannot_hide_a_padding_boundary(access):
    storage = tensor("storage", (2, 3), "int32")
    view = tensor("view", (10,), "int16")
    base = ir.Call("tirx.reinterpret", [storage.data], ty=view.data.ty)
    p = offset_pointer(base, 4)
    if access == "tensor":
        access_expr = pointer(view, [3], extent=2)
    else:
        pointer_var = ir.Var("pointer_alias", view.data.ty)
        access_expr = ir.Call("tl.ptr_access", [pointer_var, integer(3), integer(2), integer(1)])
    body = [allocate(storage), declare(view, p)]
    if access == "pointer":
        body.append(ir.Bind(pointer_var, p))
    body += [annotation({"tl.buffer_version": {storage: 2}}), ir.Evaluate(access_expr)]
    with pytest.raises(ValueError, match="crosses a 32-byte-aligned version slot"):
        apply("RewriteAscendBufferVersionLayout", body)


def test_alias_store_uses_same_mapping_as_owner_read():
    storage = tensor("storage", (2, 3), "int32")
    view = tensor("view", (5,), "int32")
    result = apply(
        "RewriteAscendBufferVersionLayout",
        [
            allocate(storage),
            declare(view, offset_pointer(storage.data, 4)),
            annotation({"tl.buffer_version": {storage: 2}}),
            ir.TensorStore(view, [2], integer(7)),
            ir.Evaluate(ir.TensorLoad(storage, [1, 0])),
        ],
    )
    binds = definitions(result)
    (store,) = objects(result, ir.TensorStore)
    (load,) = objects(result, ir.TensorLoad)
    store_owner, store_offset = address(store.dest, binds)
    load_owner, load_offset = address(load.source, binds)
    assert store_owner.same_as(load_owner)
    assert store_offset * 8 + constant(store.indices[0]) * 32 == 256
    assert load_offset * 8 + constant(load.indices[0]) * 32 == 256


@pytest.mark.parametrize("access", ["load", "store", "pointer"])
def test_wide_alias_element_cannot_span_version_padding(access):
    storage = tensor("storage", (2, 3), "int8")
    view = tensor("view", (1,), "int16")
    base = ir.Call("tirx.reinterpret", [storage.data], ty=view.data.ty)
    # The two bytes of view[0] came from compact bit offsets 16 and 24.
    # Padding maps those bytes to 16 and 256. A single contiguous int16
    # access cannot preserve both; a valid starting address is insufficient.
    if access == "load":
        use = ir.Evaluate(ir.TensorLoad(view, [0]))
    elif access == "store":
        use = ir.TensorStore(view, [0], integer(7, "int16"))
    else:
        use = ir.Evaluate(pointer(view, [0], extent=1))
    with pytest.raises(ValueError, match="crosses a 32-byte-aligned version slot"):
        apply(
            "RewriteAscendBufferVersionLayout",
            [
                allocate(storage),
                declare(view, offset_pointer(base, 2)),
                annotation({"tl.buffer_version": {storage: 2}}),
                use,
            ],
        )


@pytest.mark.parametrize("nested", [False, True])
def test_missing_pointer_type_is_rejected_before_partial_version_rewrite(nested):
    storage = tensor("storage", (2, 3), "int32")
    view = tensor("view", (10,), "int16")
    base = ir.Call("tirx.reinterpret", [storage.data], ty=view.data.ty)
    untyped = offset_pointer(base, 4, explicit_type=False)
    assert isinstance(untyped.ty, ir.MissingType)
    # The frozen Op deliberately has no inference rule. Signature validation
    # and definition/scope checks do not guarantee a fully typed input.
    assert isinstance(ir.reinfer_type(untyped), ir.MissingType)
    untyped.validate()
    address_expr = offset_pointer(untyped, 0, explicit_type=False) if nested else untyped
    function = tirx.Function(
        [],
        [
            allocate(storage),
            declare(view, address_expr),
            annotation({"tl.buffer_version": {storage: 2}}),
            ir.Evaluate(ir.TensorLoad(view, [4])),
        ],
    )
    assert tirx.analysis.check_well_formed(function)
    saved = ir.save_json(function)
    transform = tvm_ffi.get_global_func("tl.transform.RewriteAscendBufferVersionLayout")()
    with pytest.raises(ValueError, match="exact, fully typed pointer provenance"):
        transform(ir.IRModule({"main": function}))
    assert ir.save_json(function) == saved


def test_scope_changing_alias_is_rejected_before_partial_version_rewrite():
    storage = tensor("storage", (2, 3), "int32")
    view = tensor("view", (6,), "int32", scope="global")
    reinterpreted = ir.Call("tirx.reinterpret", [storage.data], ty=view.data.ty)
    with pytest.raises(ValueError, match="exact, fully typed pointer provenance"):
        apply(
            "RewriteAscendBufferVersionLayout",
            [
                allocate(storage),
                declare(view, reinterpreted),
                annotation({"tl.buffer_version": {storage: 2}}),
                ir.Evaluate(ir.TensorLoad(view, [3])),
            ],
        )


def test_effectful_alias_offset_is_rejected_before_partial_version_rewrite():
    storage = tensor("storage", (2, 3), "int32")
    view = tensor("view", (6,), "int32")
    offset = ir.Call("tirx.call_extern", [ir.StringImm("next_offset")], ty=ir.PrimType("int32"))
    p = ir.Call("tirx.ptr_byte_offset", [storage.data, offset], ty=storage.data.ty)
    with pytest.raises(ValueError, match="exact, fully typed pointer provenance"):
        apply(
            "RewriteAscendBufferVersionLayout",
            [
                allocate(storage),
                declare(view, p),
                annotation({"tl.buffer_version": {storage: 2}}),
                ir.Evaluate(ir.TensorLoad(view, [3])),
            ],
        )


def test_dynamic_alias_offset_has_same_finite_bit_oracle():
    storage = tensor("storage", (2, 3), "int32")
    view = tensor("view", (6,), "int16")
    start, index = ir.Var("start", "int32"), ir.Var("index", "int32")
    base = ir.Call("tirx.reinterpret", [storage.data], ty=view.data.ty)
    p = ir.Call("tirx.ptr_byte_offset", [base, start * 2], ty=base.ty)
    result = apply(
        "RewriteAscendBufferVersionLayout",
        [
            allocate(storage),
            declare(view, p),
            annotation({"tl.buffer_version": {storage: 2}}),
            ir.Evaluate(ir.TensorLoad(view, [index])),
        ],
        [start, index],
    )
    (load,) = objects(result, ir.TensorLoad)
    owner, offset = address(load.source, definitions(result))
    assert owner.name == "storage" and offset == 0
    for begin in range(6):
        for element in range(6):
            analyzer = sym.Analyzer()
            actual = analyzer.simplify(substitute(load.indices[0], {start: begin, index: element}))
            assert int(actual.value) * 16 == original_physical_bit((begin + element) * 16)
