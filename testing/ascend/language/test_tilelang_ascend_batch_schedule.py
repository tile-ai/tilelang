"""Batch scheduling survives splitting and is local to each device function."""

import re

import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang import tvm


def _two_kernels():
    @T.prim_func
    def main(out: T.Tensor((2,), "int32")):
        with T.Kernel(1):
            out[0] = 1
        with T.Kernel(1):
            out[1] = 2

    return main


@tilelang.testing.requires_ascend(support_required="compile-only")
@pytest.mark.parametrize("enabled", [None, False, True])
def test_batch_schedule_survives_host_device_split(enabled):
    func = _two_kernels()
    if enabled is not None:
        func = func.with_attr("ascend.batch_schedule", enabled)
    source = tilelang.lower(func, target="ascend").kernel_source
    assert source.count('extern "C"') == 2
    assert source.count("__schedmode__(1)") == (2 if enabled else 0)


@tilelang.testing.requires_ascend(support_required="compile-only")
@pytest.mark.parametrize("following", [(False, None), (None, False)])
def test_batch_schedule_does_not_leak_between_device_functions(following):
    # Bypass splitting here to exercise differently attributed functions in one
    # code generator, including an explicit false value and an absent value.
    mod = tvm.IRModule(
        {
            name: tvm.tirx.PrimFunc([], tvm.tirx.Evaluate(0)).with_attrs({"global_symbol": name, "tirx.is_global_func": True})
            for name in ("first", "second", "third")
        }
    )
    # Follow the actual Map iteration order, which need not be insertion order.
    order = list(mod.functions)
    enabled_by_name = {}
    for gv, enabled in zip(order, (True, *following)):
        if enabled is not None:
            mod[gv] = mod[gv].with_attr("ascend.batch_schedule", enabled)
        enabled_by_name[gv.name_hint] = bool(enabled)
    assert list(mod.functions) == order
    build = tvm.get_global_func("target.build.tilelang_ascend_without_compile")
    source = build(mod, tvm.target.Target("ascend")).inspect_source()
    for name, enabled in enabled_by_name.items():
        declaration = re.search(r'extern "C"[^\n]*\b' + name + r"\(", source)
        assert declaration is not None
        assert ("__schedmode__(1)" in declaration.group()) == enabled


if __name__ == "__main__":
    tilelang.testing.main()
