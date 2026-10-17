"""Writable GM uses scalar dcache bypass; read-only GM and VF accesses do not."""

import pytest
from tilelang import tvm
from tilelang.ascend import transform
from tvm import tirx
from testing.ascend._ir import calls, nodes


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
        write = tirx.Evaluate(tirx.Call("void", tvm.ir.Op.get(op), [writable.access_ptr("w"), local.access_ptr("r")]))
    # Put the read first: writability is a property of the entire function.
    before = tvm.IRModule({"main": tirx.PrimFunc([writable.data, readonly.data], tirx.SeqStmt([read, write]))})
    after = transform.MarkScalarDcacheBypass()(before)
    (bypass,) = calls(after, "tl.ascend_read_gm_bypass_dcache")
    assert bypass.args[0].args[0].buffer.same_as(writable)
    writes = calls(after, "tl.ascend_write_gm_bypass_dcache")
    assert len(writes) == (1 if writer == "scalar" else 0)
    if writes:
        assert writes[0].args[0].args[0].buffer.same_as(writable)
    assert any(load.buffer.same_as(readonly) for load in nodes(after, tirx.BufferLoad))
    assert any(store.buffer.same_as(local) for store in nodes(after, tirx.BufferStore))


@pytest.mark.parametrize("kind", ["SIMT_VF", "SIMD_VF"])
def test_vf_memory_operations_keep_their_own_access_path(kind):
    buffer = tirx.decl_buffer((64,), "float32", name="buffer")
    body = tirx.BufferStore(buffer, buffer[0] + tirx.const(1, "float32"), [0])
    before = tvm.IRModule({"main": tirx.PrimFunc([buffer.data], tirx.SBlock([], [], [], kind, body))})
    after = transform.MarkScalarDcacheBypass()(before)
    tvm.ir.assert_structural_equal(after, before)
