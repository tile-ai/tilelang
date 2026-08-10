"""Host codegen component types and shared implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from tvm import IRModule
from tvm.target import Target

from tilelang import tvm
from tilelang.utils.pass_events import run_codegen_with_instrumentation

HostCodegenFunc = Callable[[IRModule, Target], IRModule]
HostCodegenHookFunc = Callable[[IRModule, Target, Target], IRModule]


def global_func_host_codegen(global_func_name: str) -> HostCodegenFunc:
    """Create a host codegen callback backed by a TVM global function."""

    def build(mod: IRModule, target_host: Target) -> IRModule:
        return run_codegen_with_instrumentation(
            global_func_name,
            mod,
            target_host,
            lambda: tvm.ffi.get_global_func(global_func_name)(mod, target_host),
        )

    return build


@dataclass(frozen=True, slots=True)
class HostCodegen:
    """Host codegen entry point for one host target variant."""

    name: str
    build: HostCodegenFunc

    def lower(self, mod: IRModule, target_host: Target) -> IRModule:
        return self.build(mod, target_host)


@dataclass(frozen=True, slots=True)
class HostCodegenHook:
    """Device-backend hook applied before host codegen build."""

    name: str
    apply: HostCodegenHookFunc

    def lower(self, mod: IRModule, target_host: Target, target: Target) -> IRModule:
        return self.apply(mod, target_host, target)


STANDARD_HOST_CODEGENS: Mapping[str, HostCodegen] = MappingProxyType(
    {
        "c": HostCodegen("c", build=global_func_host_codegen("target.build.tilelang_c_host")),
        "llvm": HostCodegen("llvm", build=global_func_host_codegen("target.build.llvm")),
    }
)
