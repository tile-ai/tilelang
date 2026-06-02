from __future__ import annotations

from tilelang import tvm
from tvm import tirx
from tvm.target import Target

import tilelang.transform


def prepare_device_codegen(device_mod: tvm.IRModule) -> tvm.IRModule:
    """Run common cleanup before backend device codegen."""

    device_mod = tilelang.transform.LowerIntrin()(device_mod)
    device_mod = tirx.transform.Simplify()(device_mod)
    device_mod = tilelang.transform.HoistBroadcastValues()(device_mod)
    return device_mod


def build_device_with_global_func(device_mod: tvm.IRModule, target: Target, global_func_name: str) -> tvm.IRModule:
    device_mod = prepare_device_codegen(device_mod)
    return tvm.ffi.get_global_func(global_func_name)(device_mod, target)
