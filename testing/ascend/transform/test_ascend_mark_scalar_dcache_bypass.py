import tilelang.ascend.language as T

"""Writable GM uses scalar dcache bypass; read-only GM and VF accesses do not."""

from tvm_tirx import ir
from testing.ascend._ir import access_tensor, sblock, tensor

import pytest
import tvm_tirx
from tilelang.ascend import transform
from tvm_tirx import tirx
from testing.ascend._ir import calls, nodes


@pytest.mark.parametrize("writer", ["scalar", "mte3", "fixpipe"])
def test_only_writable_global_storage_bypasses_dcache(writer):
    writable = tensor((64,), "float32", name="writable")
    readonly = tensor((64,), "float32", name="readonly")
    local = tensor((64,), "float32", name="local", scope="shared.dyn")
    read = ir.TensorStore(local, [0], writable[0] + readonly[0])
    if writer == "scalar":
        write = ir.TensorStore(writable, [0], local[0])
    else:
        op = "tl.ascend_copy_ubuf_to_gm" if writer == "mte3" else "tl.ascend_copy_matrix_cc_to_gm"
        # The pass runs on lowered DMA intrinsics. The destination access_ptr
        # is the write footprint; transfer geometry is irrelevant here.
        write = ir.Evaluate(ir.Call(tvm_tirx.ir.Op.get(op), [T.access_ptr(writable, "w"), T.access_ptr(local, "r")], ty="void"))
    # Put the read first: writability is a property of the entire function.
    before = tvm_tirx.IRModule({"main": tirx.Function([writable, readonly], ir.SeqStmt([read, write]))})
    after = transform.MarkScalarDcacheBypass()(before)
    (bypass,) = calls(after, "tl.ascend_read_gm_bypass_dcache")
    assert access_tensor(bypass.args[0].args[0]).same_as(writable)
    writes = calls(after, "tl.ascend_write_gm_bypass_dcache")
    assert len(writes) == (1 if writer == "scalar" else 0)
    if writes:
        assert access_tensor(writes[0].args[0].args[0]).same_as(writable)
    assert any(access_tensor(load).same_as(readonly) for load in nodes(after, ir.TensorLoad))
    assert any(access_tensor(store).same_as(local) for store in nodes(after, ir.TensorStore))


@pytest.mark.parametrize("kind", ["SIMT_VF", "SIMD_VF"])
def test_vf_memory_operations_keep_their_own_access_path(kind):
    buffer = tensor((64,), "float32", name="buffer")
    body = ir.TensorStore(buffer, [0], buffer[0] + ir.const(1, "float32"))
    before = tvm_tirx.IRModule({"main": tirx.Function([buffer], sblock(kind, body))})
    after = transform.MarkScalarDcacheBypass()(before)
    tvm_tirx.ir.assert_structural_equal(after, before)
