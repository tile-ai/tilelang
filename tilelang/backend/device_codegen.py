"""Device codegen component types and helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tvm import IRModule
from tvm.target import Target

from tilelang import tvm
from tilelang.utils.pass_events import run_codegen_with_instrumentation

DeviceCodegenFunc = Callable[[IRModule, Target], IRModule]


def global_func_device_codegen(global_func_name: str) -> DeviceCodegenFunc:
    """Create a device codegen callback backed by a TVM global function."""

    def build(mod: IRModule, target: Target) -> IRModule:
        return run_codegen_with_instrumentation(
            global_func_name,
            mod,
            target,
            lambda: tvm.ffi.get_global_func(global_func_name)(mod, target),
        )

    return build


@dataclass(frozen=True, slots=True)
class DeviceCodegen:
    """Device codegen entry points for one backend target variant."""

    name: str
    build: DeviceCodegenFunc | None = None
    build_without_compile: DeviceCodegenFunc | None = None

    def lower(self, mod: IRModule, target: Target, *, compile_device: bool) -> IRModule:
        build_func = self.build if compile_device else self.build_without_compile
        if build_func is None:
            mode = "with compilation" if compile_device else "without compilation"
            raise ValueError(f"Device codegen '{self.name}' for target '{target.kind.name}' does not support lowering {mode}")
        return build_func(mod, target)
