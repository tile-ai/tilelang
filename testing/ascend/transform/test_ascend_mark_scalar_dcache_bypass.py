"""Writable GM uses scalar dcache bypass; read-only GM and VF accesses do not."""

import pytest
from tvm_tirx import ir
from tilelang import tvm
from tilelang.ascend import transform
from tilelang._compat import tirx
from testing.ascend._ir import calls, nodes, kernel
from tilelang.ascend import language as T
from tilelang._compat.tensor import TensorView


@pytest.mark.parametrize("writer", ["scalar", "mte3", "fixpipe"])
def test_only_writable_global_storage_bypasses_dcache(writer):
    writable = tirx.decl_buffer((64,), "float32", name="writable")
    readonly = tirx.decl_buffer((64,), "float32", name="readonly")
    local = tirx.decl_buffer((64,), "float32", name="local", scope="shared.dyn")
    read = tirx.BufferStore(local, writable[0] + readonly[0], [0])
    if writer == "scalar":
        write = tirx.BufferStore(writable, local[0], [0])
    else:
        op = "tl.ascend_copy_ubuf_to_gm" if writer == "mte3" else "tl.ascend_copy_matrix_cc_to_gm"
        # The pass runs on lowered DMA intrinsics. The destination access_ptr
        # is the write footprint; transfer geometry is irrelevant here.
        write = ir.Evaluate(ir.Call(op, [T.access_ptr(writable[0], "w", extent=64), T.access_ptr(local[0], "r", extent=64)], ty="void"))
    # Put the read first: writability is a property of the entire function.
    before = kernel([read, write], buffers=[local], params=[writable, readonly])
    after = transform.MarkScalarDcacheBypass()(before)
    (bypass,) = calls(after, "tl.ascend_read_gm_bypass_dcache")
    assert bypass.args[0].args[0].source.same_as(writable)
    writes = calls(after, "tl.ascend_write_gm_bypass_dcache")
    assert len(writes) == (1 if writer == "scalar" else 0)
    if writes:
        assert writes[0].args[0].args[0].source.same_as(writable)
    assert any(load.source.same_as(readonly) for load in nodes(after, tirx.BufferLoad))
    assert any(store.dest.same_as(local) for store in nodes(after, tirx.BufferStore))


@pytest.mark.parametrize("alias_kind", ["tensor", "pointer", "offset"])
def test_alias_writes_mark_the_original_global_tensor(alias_kind):
    original = TensorView.create("original", [64], "float32")
    alias = TensorView.create("alias", [64], "float32")
    local = TensorView.create("local", [1], "float32", scope="shared.dyn")
    bindings = [ir.Bind(alias.tensor, original.tensor)]
    if alias_kind != "tensor":
        pointer = ir.Var("pointer", original.data.ty)
        address = original.data if alias_kind == "pointer" else tirx.ptr_byte_offset(original.data, 0, ty=original.data.ty)
        bindings = [ir.Bind(pointer, address), alias.declare(pointer)]
    before = kernel(
        [*bindings, local.store(0, original.load(0)), alias.store(0, local.load(0))], buffers=[local.tensor], params=[original.tensor]
    )
    after = transform.MarkScalarDcacheBypass()(before)
    assert len(calls(after, "tl.ascend_read_gm_bypass_dcache")) == 1
    assert len(calls(after, "tl.ascend_write_gm_bypass_dcache")) == 1


def test_access_pointer_preserves_its_marker_and_rewrites_live_index_loads():
    values = TensorView.create("values", [64], "float32")
    indices = TensorView.create("indices", [1], "int32")
    pointer = T.access_ptr(values.load(indices.load(0)), "r", extent=1)
    before = kernel(
        [indices.store(0, 3), values.store(0, 1), ir.Evaluate(tirx.call_extern("int32", "consume", pointer))],
        params=[values.tensor, indices.tensor],
    )
    after = transform.MarkScalarDcacheBypass()(before)
    (address,) = calls(after, "tl.access_ptr")
    assert isinstance(address.args[0], ir.TensorLoad)
    assert address.args[0].source.same_as(values.tensor)
    assert address.args[0].indices[0].op.name == "tl.ascend_read_gm_bypass_dcache"


@pytest.mark.parametrize("kind", ["SIMT_VF", "SIMD_VF"])
def test_vf_memory_operations_keep_their_own_access_path(kind):
    buffer = tirx.decl_buffer((64,), "float32", name="buffer")
    body = tirx.BufferStore(buffer, buffer[0] + tirx.const(1, "float32"), [0])
    before = tvm.IRModule(
        {"main": tirx.PrimFunc([buffer], ir.RegionStmt("tl.scope", [], [], {"name_hint": kind}, body), ret_type=ir.TupleType([]))}
    )
    after = transform.MarkScalarDcacheBypass()(before)
    tvm.ir.assert_structural_equal(after, before)
