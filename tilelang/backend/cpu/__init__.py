from __future__ import annotations

from tilelang.backend.base import DeviceBackend

from .execution import CPU_DEFAULT_EXECUTION_BACKEND, CPU_EXECUTION_BACKENDS
from .ffi import C_SOURCE_BUILDER, LLVM_SOURCE_BUILDER
from .passes import CpuPassHooks
from .target import is_c_target, is_llvm_target


def get_backends() -> tuple[DeviceBackend, ...]:
    hooks = CpuPassHooks()
    return (
        DeviceBackend(
            name="cpu-c",
            family="cpu",
            match_target=is_c_target,
            source_builder=C_SOURCE_BUILDER,
            compiled_builder=None,
            execution_backends=CPU_EXECUTION_BACKENDS,
            default_execution_backend=CPU_DEFAULT_EXECUTION_BACKEND,
            source_kind="c",
            pass_hooks=hooks,
            metadata={"dialect": "c"},
        ),
        DeviceBackend(
            name="cpu-llvm",
            family="cpu",
            match_target=is_llvm_target,
            source_builder=LLVM_SOURCE_BUILDER,
            compiled_builder=None,
            execution_backends=CPU_EXECUTION_BACKENDS,
            default_execution_backend=CPU_DEFAULT_EXECUTION_BACKEND,
            source_kind="llvm",
            pass_hooks=hooks,
            metadata={"dialect": "llvm"},
        ),
    )

