"""Inspect native memory coordinates through declarations and typed pointers.

Tests compare these actual coordinates with independent expected geometry.
Unsupported pointer forms fail explicitly; no IR-print parsing is used.
"""

import tvm_tirx
from tvm_tirx import ir, tirx

from testing.ascend._ir import nodes, scheduled_fields


def _i64(value):
    if isinstance(value, int):
        return ir.const(value, "int64")
    return value if str(value.ty) == "int64" else ir.prim.Cast(ir.PrimType("int64"), value)


def _element_bits(ty):
    if isinstance(ty, ir.PtrType):
        ty = ty.element_type
    elif isinstance(ty, tirx.TensorType):
        ty = ty.dtype
    dtype = ty.dtype
    return (8 if str(dtype).startswith("bool") else dtype.bits) * dtype.lanes


def address_bits(program, expression):
    """Return allocation/parameter identity and a bit offset within that owner."""
    bindings = {}
    for binding in nodes(program, ir.Bind):
        value = binding.value
        fields = scheduled_fields(binding)
        if fields is not None:
            # The marker has one value operand. Static stage/task metadata is
            # in attrs; dynamic metadata adds a validated trailing Tuple.
            # The surrounding Bind still owns the allocation/view identity.
            assert set(fields) == {"stage", "task"}
            value.validate()
            tvm_tirx.ir.assert_structural_equal(value.ty, value.args[0].ty)
            value = value.args[0]
        bindings[binding.var] = value
    visiting = set()
    analyzer = tvm_tirx.sym.Analyzer()

    def tensor_access(tensor, indices):
        owner, base = walk(tensor)
        ty = tensor.ty
        assert isinstance(ty, tirx.TensorType)
        if ty.layout is not None:
            coordinates = ty.layout.apply(*indices, shape=list(ty.shape))
            assert set(coordinates) == {"m"}, "Address oracle requires a memory-only layout"
            index = coordinates["m"]
        else:
            assert len(indices) == len(ty.shape)
            index = ir.const(0, "int64")
            for value, extent in zip(indices, ty.shape):
                index = index * _i64(extent) + _i64(value)
        return owner, base + _i64(index) * _element_bits(ty)

    def walk(value):
        if isinstance(value, ir.Var):
            assert isinstance(value.ty, (ir.PtrType, tirx.TensorType))
            assert value not in visiting, "Cyclic native pointer definition"
            binding = bindings.get(value)
            if binding is None:
                return value, ir.const(0, "int64")
            if isinstance(binding, ir.Call) and binding.op.name == "tirx.alloc_tensor":
                return value, ir.const(0, "int64")
            visiting.add(value)
            if isinstance(binding, ir.Call) and binding.op.name == "tirx.decl_tensor":
                result = walk(binding.args[0])
            else:
                result = walk(binding)
            visiting.remove(value)
            return result
        if isinstance(value, ir.TensorStore):
            return tensor_access(value.dest, value.indices)
        if isinstance(value, ir.TensorLoad):
            return tensor_access(value.source, value.indices)
        if isinstance(value, ir.Call):
            op = value.op.name
            if op in {"tirx.tensor_data_ptr", "tirx.reinterpret"}:
                return walk(value.args[0])
            if op in {"tl.access_ptr", "tirx.address_of"}:
                assert isinstance(value.args[0], ir.TensorLoad)
                return walk(value.args[0])
            if op in {"tl.ptr_access", "tirx.ptr_byte_offset", "tirx.handle_add_byte_offset"}:
                owner, offset = walk(value.args[0])
                unit = _element_bits(value.args[0].ty) if op == "tl.ptr_access" else 8
                return owner, offset + _i64(value.args[1]) * unit
        raise AssertionError(f"Unsupported native address form: {type(value).__name__}")

    owner, offset = walk(expression)
    return owner, analyzer.simplify(offset)


def element_offset(program, expression):
    """Address in logical elements of the actual access/pointer type."""
    _, bits = address_bits(program, expression)
    ty = expression.dest.ty if isinstance(expression, ir.TensorStore) else expression.ty
    width = _element_bits(ty)
    analyzer = tvm_tirx.sym.Analyzer()
    assert analyzer.can_prove_equal(bits % width, 0), "Nonintegral element address"
    return analyzer.simplify(bits // width)


def storage_owner(program, expression):
    return address_bits(program, expression)[0]
