"""Typed pointer consumers distinguish storage escapes from scalar address inputs."""

import pytest
import tilelang.ascend
from tilelang.ascend import transform
from tvm_tirx import ir, tirx
from testing.ascend._ir import annotations, kernel, seq, tensor
from testing.ascend.auto_schedule._scheduled_ir import unit


@pytest.mark.parametrize(
    "consumer,may_alias",
    [("address_of", False), ("carrier", False), ("returned", False), ("bounded", True), ("scalar_argument", True)],
)
def test_typed_pointer_consumer_preserves_storage_liveness(consumer, may_alias):
    tilelang.ascend.register_backend()
    a = tensor((8,), "int32", name="a", scope="shared.dyn")
    b = tensor((8,), "int32", name="b", scope="shared.dyn")
    out = tensor((2,), "int32", name="out")
    carrier = tensor((1,), "uint64", name="carrier", scope="local.var")
    pointer_type = ir.PtrType(ir.PrimType("int32"), "shared.dyn")
    pointer = ir.Call("tl.access_ptr", [ir.TensorLoad(a, [0]), ir.const(1), ir.const(1)], ty=pointer_type)
    final = []
    if consumer == "address_of":
        pointer = tirx.address_of(ir.TensorLoad(a, [0]))
    elif consumer == "carrier":
        final.append(ir.TensorStore(carrier, [0], ir.Call("tirx.reinterpret", [pointer], ty=ir.PrimType("uint64"))))
        pointer = ir.Call("tirx.reinterpret", [ir.TensorLoad(carrier, [0])], ty=pointer_type)
    elif consumer == "returned":
        pointer = tirx.call_extern("advance", pointer, ty=pointer_type)
    elif consumer == "scalar_argument":
        # a[0] is a precise scalar input to this address calculation. The
        # returned pointer does not carry the storage address of a.
        pointer = tirx.call_extern(
            "from_scalar_offset",
            ir.TensorLoad(a, [0]),
            ty=ir.PtrType(ir.PrimType("int32"), "global"),
        )
    value = tirx.call_extern("read_value", pointer, ty="int32") + ir.TensorLoad(a, [0])
    final.append(ir.TensorStore(out, [0], value))
    before = kernel(
        seq(
            unit(ir.TensorStore(a, [1], ir.const(100))),
            unit(ir.TensorStore(b, [1], ir.const(200))),
            unit(ir.TensorStore(out, [1], ir.TensorLoad(b, [1]))),
            unit(ir.TensorStore(a, [0], ir.const(300))),
            unit(seq(*final)),
        ),
        buffers=[a, b] + ([carrier] if consumer == "carrier" else []),
        params=[out],
        allocation_stage=0,
    )
    after = transform.InsertSync()(before)
    pairs = {
        tuple(sorted((storage.name, other.name)))
        for mapping in annotations(after)
        for storage, compatible in mapping.get("tl.buffer_alias_map", {}).items()
        for other in compatible
    }
    assert (("a", "b") in pairs) == may_alias
