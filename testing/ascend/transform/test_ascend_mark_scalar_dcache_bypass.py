"""Writable GM uses scalar dcache bypass; read-only GM and VF accesses do not."""

import re

import pytest
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang.ascend import transform
from tilelang.engine.lower import lower
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


def _scalar_write_kernel():
    @T.prim_func
    def func(
        rw_buf: T.Buffer((16,), "float32"),
        ro_buf: T.Buffer((16,), "float32"),
    ):
        with T.Kernel(1) as _:
            rw_buf[0] = T.float32(1)
            a = rw_buf[0]
            b = ro_buf[0]
            rw_buf[0] = a + b

    return func


def _mte_write_kernel():
    @T.prim_func
    def func(
        rw_buf: T.Buffer((16,), "float32"),
        ro_buf: T.Buffer((16,), "float32"),
    ):
        with T.Kernel(1) as _:
            ub = T.alloc_shared((16,), "float32")
            T.copy(ro_buf, ub)
            a = rw_buf[0]
            ub[0] = a + T.float32(1)
            T.copy(ub, rw_buf)

    return func


def _bypass_call_exprs(target):
    if target == "pto":
        return "_tl_pto_read_gm_bypass_dcache(", "_tl_pto_write_gm_bypass_dcache("
    return "tl::read_gm_bypass_dcache(", "tl::write_gm_bypass_dcache("


def _bypass_call_pattern(target):
    if target == "pto":
        return r"_tl_pto_(?:read|write)_gm_bypass_dcache\([^;\n]+\)"
    return r"tl::(?:read|write)_gm_bypass_dcache\([^;]+\)"


def _assert_bypass_calls_only_on_writable(source, target):
    bypass_calls = re.findall(_bypass_call_pattern(target), source)
    assert bypass_calls
    assert all("rw_buf" in call and "ro_buf" not in call for call in bypass_calls)


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_dcache_bypass_codegen_for_scalar_write(target):
    source = lower(_scalar_write_kernel(), target=target).kernel_source
    read_call, write_call = _bypass_call_exprs(target)
    assert read_call in source
    assert write_call in source
    assert "#include <tl_templates/ascend/dcache_bypass.h>" in source
    _assert_bypass_calls_only_on_writable(source, target)
    if target == "pto":
        assert any("ro_buf" in call for call in re.findall(r"scalar\.load\([^;\n]+\)", source))
    else:
        assert "ro_buf[0]" in source


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_dcache_bypass_codegen_for_mte_write(target):
    source = lower(_mte_write_kernel(), target=target).kernel_source
    read_call, _ = _bypass_call_exprs(target)
    assert read_call in source
    _assert_bypass_calls_only_on_writable(source, target)
