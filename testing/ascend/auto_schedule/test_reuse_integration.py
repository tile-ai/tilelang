"""The reuse contract survives scheduled lowering and is consumed by packing."""

import tilelang
from tvm_tirx import ir
import tilelang.ascend.language as T
from tilelang import tvm
from tilelang._compat import tirx
from testing.ascend._ir import allocated_buffers, nodes
from testing.ascend.auto_schedule._reuse_utils import _make_cyclic_scalar_program


def test_alias_contract_lifetime():
    snapshots = {}
    wanted = {
        "tl.InsertSync",
        "tl.LowerScheduledTIR",
        "tl.MergeUBAllocations",
    }

    @tvm.transform.instrument.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod

    with tvm.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        tilelang.lower(_make_cyclic_scalar_program(), target="ascend")

    assert any("tl.buffer_alias_map" in region.attrs for region in nodes(snapshots["tl.InsertSync"], ir.RegionStmt))
    assert any("tl.buffer_alias_map" in region.attrs for region in nodes(snapshots["tl.LowerScheduledTIR"], ir.RegionStmt))
    assert not any("tl.buffer_alias_map" in region.attrs for region in nodes(snapshots["tl.MergeUBAllocations"], ir.RegionStmt))


def test_fixed_versions_allow_ub_reuse_above_raw_capacity():
    @T.prim_func
    def main(x: T.Tensor((8,), "int32"), out: T.Tensor((8, 2), "int32")):
        with T.Kernel(1):
            a = T.alloc_shared((32768,), "int32")
            b = T.alloc_shared((32768,), "int32")
            T.annotate_buffer_versions({a: 1, b: 1})
            for i in T.Pipelined(8, num_stages=1):
                with T.Stage(0):
                    a[i * 4096] = x[i]
                with T.Stage(0):
                    out[i, 0] = a[i * 4096]
                with T.Stage(0):
                    b[i * 4096] = x[i] + 1
                with T.Stage(0):
                    out[i, 1] = b[i * 4096]

    allocations = []

    @tvm.transform.instrument.pass_instrument
    class CaptureAllocations:
        def run_after_pass(self, mod, info):
            if info.name == "tl.MergeUBAllocations":
                allocations.extend(
                    tensor for func in mod.functions.values() for tensor in allocated_buffers(func) if tensor.scope().name == "shared.dyn"
                )

    with tvm.transform.PassContext(instruments=[CaptureAllocations()]):
        tilelang.lower(main, target="ascend", enable_device_compile=False)

    # The raw sum is 256 KiB, but disjoint lifetimes need only 128 KiB of UB.
    (allocation,) = allocations
    assert allocation.dtype == "uint8"
    assert tuple(int(dim) for dim in allocation.shape) == (131072,)
