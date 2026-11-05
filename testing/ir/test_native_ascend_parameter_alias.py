"""Only explicit parameter alias contracts can remove cross-core hazards."""

import pytest
from tilelang.ascend import transform
from tvm_tirx import ir

from test_native_ascend_core_passes import _apply
from test_native_ascend_core_resolution import _kernel, _unit
from test_native_ascend_ops import function
from test_native_ascend_schedule_units import _bind_target
from test_native_cuda_copy import calls
from test_native_reducer_contracts import allocate, annotation, attribute, integer, region, tensor, tileop


def _program(origin="tensor", contract=True, exception=None):
    left = tensor("left", [16], "float32", "global")
    right = tensor("right", [16], "float32", "global")
    first = tensor("first", [16], "float32", "shared.dyn")
    second = tensor("second", [16], "float32", "shared.dyn")
    definitions = [_unit(allocate(first), 1), _unit(allocate(second), 4)]
    params = [left, right]
    if origin != "tensor":
        if origin == "pointer":
            roots = [ir.Var(name, ir.PtrType("float32", "global")) for name in ("input", "output")]
            pointers = roots
        else:
            roots = [tensor(name, [32], "float32", "global") for name in ("input", "output")]
            if origin == "same-root":
                roots = roots[:1]
            pointers = [roots[0].data, roots[-1].data]
            if origin == "views":
                pointers = [ir.Call("tirx.ptr_byte_offset", [pointer, integer(16, "int64")], ty=pointer.ty) for pointer in pointers]
            elif origin == "scope-roundtrip":
                shared = ir.Call("tirx.reinterpret", [pointers[0]], ty=ir.PtrType("float32", "shared.dyn"))
                pointers[0] = ir.Call("tirx.reinterpret", [shared], ty=pointers[0].ty)
        params = roots
        definitions.extend(
            _unit(ir.Bind(view, ir.Call("tirx.decl_tensor", [pointer], ty_args=[view.ty])), core)
            for view, pointer, core in zip((left, right), pointers, (1, 4))
        )
    if exception == "statement":
        definitions.append(annotation({"tl.non_restrict_params": [left]}))
    elif exception == "unknown":
        address = ir.Var("unknown_address", "uint64")
        params = [*params, address]
        pointer = ir.Call("tirx.reinterpret", [address], ty=ir.PtrType("float32", "global"))
        definitions.append(annotation({"tl.non_restrict_params": [pointer]}))
    tasks = [
        _unit(tileop("ascend_copy", region(left), region(first)), 1),
        _unit(tileop("ascend_copy", region(second), region(right)), 4),
    ]
    value = _kernel(params, [*definitions, *tasks])
    if exception == "kernel":
        value = function(params, [attribute(None, "tl.kernel", {"tl.non_restrict_params": [params[0]]}, value.body)])
    elif exception == "region":
        value = function(params, [attribute(None, "tl.non_restrict_params", [params[0]], value.body)])
    if contract is not None:
        value = value.with_attr("tirx.noalias", contract)
    if exception == "function":
        value = value.with_attr("tl.non_restrict_params", [params[0]])
    elif exception == "index":
        value = value.with_attr("tl.may_alias_param_indices", [0])
    return _bind_target(value)


@pytest.mark.parametrize("origin", ["tensor", "pointer", "views"])
@pytest.mark.parametrize("contract", [None, False, True], ids=["unspecified", "may-alias", "noalias"])
def test_external_parameters_use_only_the_existing_function_contract(origin, contract):
    value = _program(origin, contract)
    if contract is not True:
        with pytest.raises(RuntimeError, match="does not support synchronization between AIV0 and AIV1"):
            _apply(transform.InsertSync(), value)
        return
    result = _apply(transform.InsertSync(), value)
    assert len(calls(result, "tl.tileop.ascend_copy")) == 2
    assert not calls(result, "tl.ascend_cross_core_set_flag")


@pytest.mark.parametrize("origin", ["tensor", "pointer", "views"])
@pytest.mark.parametrize("exception", ["function", "index", "statement", "kernel", "region", "unknown"])
def test_nonrestrict_exceptions_preserve_possible_cross_core_hazards(origin, exception):
    value = _program(origin, True, exception)
    with pytest.raises(RuntimeError, match="does not support synchronization between AIV0 and AIV1"):
        _apply(transform.InsertSync(), value)


@pytest.mark.parametrize("origin", ["same-root", "scope-roundtrip"])
def test_noalias_does_not_separate_shared_or_opaque_storage_origins(origin):
    value = _program(origin)
    with pytest.raises(RuntimeError, match="does not support synchronization between AIV0 and AIV1"):
        _apply(transform.InsertSync(), value)
