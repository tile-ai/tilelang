"""The reuse contract survives scheduled lowering and is consumed by packing."""

from tvm_tirx import ir
from testing.ascend._ir import access_tensor, allocations, annotations

import tilelang
import tilelang.ascend.language as T
import tvm_tirx
from testing.ascend.auto_schedule._reuse_utils import _make_cyclic_scalar_program


def test_alias_contract_lifetime():
    snapshots = {}
    wanted = {
        "tl.InsertSync",
        "tl.LowerScheduledTIR",
        "tl.MergeUBAllocations",
    }

    @ir.transform.pass_instrument
    class CapturePassIR:
        def run_after_pass(self, mod, info):
            if info.name in wanted:
                snapshots[info.name] = mod

    with tvm_tirx.transform.PassContext(opt_level=3, instruments=[CapturePassIR()]):
        tilelang.lower(_make_cyclic_scalar_program(), target="ascend")

    assert any("tl.buffer_alias_map" in fields for fields in annotations(snapshots["tl.InsertSync"]))
    assert any("tl.buffer_alias_map" in fields for fields in annotations(snapshots["tl.LowerScheduledTIR"]))
    assert not any("tl.buffer_alias_map" in fields for fields in annotations(snapshots["tl.MergeUBAllocations"]))


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

    merged_allocations = []

    @ir.transform.pass_instrument
    class CaptureAllocations:
        def run_after_pass(self, mod, info):
            if info.name == "tl.MergeUBAllocations":
                merged_allocations.extend(
                    access_tensor(node)
                    for func in mod.functions.values()
                    for node in allocations(func)
                    if access_tensor(node).ty.scope.name == "shared.dyn"
                )

    with tvm_tirx.transform.PassContext(instruments=[CaptureAllocations()]):
        tilelang.lower(main, target="ascend", enable_device_compile=False)

    # The raw sum is 256 KiB, but disjoint lifetimes need only 128 KiB of UB.
    (allocation,) = merged_allocations
    assert str(allocation.ty.dtype.dtype) == "uint8"
    assert tuple(int(dim) for dim in allocation.ty.shape) == (131072,)
