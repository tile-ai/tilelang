"""Source adapter entry points over the compiler's native host graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tvm_tirx import IRModule
from tvm_tirx.target import Target

from .source_host import DeviceSourceWrapper
from .utils import (
    is_ascend_target,
    is_cpu_target,
    is_cuda_target,
    is_cutedsl_target,
    is_hip_target,
    is_metal_target,
)


class BaseWrapper(ABC):
    @abstractmethod
    def wrap(self, *args, **kwargs):
        raise NotImplementedError


def _require_lowered_modules(
    device_mod: IRModule | None,
    host_mod: IRModule | None,
) -> tuple[IRModule, IRModule]:
    """Require adapter metadata to come from the compiler's lowering result."""
    missing = [name for name, mod in (("device_mod", device_mod), ("host_mod", host_mod)) if mod is None]
    if missing:
        raise ValueError(f"Adapter source generation requires pre-lowered device_mod and host_mod; missing: {', '.join(missing)}.")
    return device_mod, host_mod


class TLCUDASourceWrapper(DeviceSourceWrapper):
    """Generate CUDA host launches from native Function call operands."""


class TLHIPSourceWrapper(DeviceSourceWrapper):
    """Generate HIP host launches from native Function call operands."""


class TLAscendSourceWrapper(DeviceSourceWrapper):
    """Generate Ascend host launches from native Function call operands."""


class TLCPUSourceWrapper:
    """Append an actual native host Function to the emitted CPU translation unit."""

    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        target: Target,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        from tilelang.jit.adapter.source_host import NATIVE_SOURCE_INIT, SourceHostProgram
        from tilelang.utils.language import retrieve_func_from_module

        self.mod = scheduled_ir_module
        self.target = target
        self.source = source
        self.pass_configs = pass_configs
        self.device_mod, self.host_mod = _require_lowered_modules(device_mod, host_mod)
        self.prim_func = retrieve_func_from_module(scheduled_ir_module)
        self.program = SourceHostProgram(self.prim_func, self.host_mod)
        self.srcpath = None
        self.libpath = None
        # Retain the already emitted device source, including source callbacks
        # and external C snippets. The fragment shares its runtime declarations.
        self.lib_code = source + NATIVE_SOURCE_INIT + self.program.emit(include_preamble=False)


class TLMetalSourceWrapper:
    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        target: Target,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        self.mod = scheduled_ir_module
        self.target = target
        self.source = source
        self.pass_configs = pass_configs
        self.device_mod = device_mod
        self.host_mod = host_mod
        self.lib_code = self.update_lib_code(source)

    def update_lib_code(self, code: str):
        self.lib_code = code
        return self.lib_code


class TLWrapper(BaseWrapper):
    """
    A wrapper class for the TileLang backend.
    """

    device_mod: IRModule | None = None
    host_mod: IRModule | None = None
    pass_configs: dict[str, Any] | None = None
    target: Target | None = None
    lib: object | None = None

    def __init__(self, target: Target):
        super().__init__()
        self.scheduled_ir_module = None
        self.pass_configs = None
        self.target = target
        self.lib = None

    def assign_optimized_module(self, scheduled_ir_module: IRModule):
        self.scheduled_ir_module = scheduled_ir_module

    def assign_pass_configs(self, pass_configs: dict[str, Any]):
        self.pass_configs = pass_configs

    def assign_host_module(self, host_mod: IRModule):
        self.host_mod = host_mod

    def assign_device_module(self, device_mod: IRModule):
        self.device_mod = device_mod

    # Get Scheduled Rt Module and return source to be compiled
    def wrap(self, c_source: str):
        assert self.scheduled_ir_module is not None, "Please assign optimized module first."
        if is_cuda_target(self.target) or is_hip_target(self.target) or is_ascend_target(self.target):
            from tilelang.jit.adapter.source_host import DeviceSourceWrapper

            wrapper_class = DeviceSourceWrapper
        elif is_cpu_target(self.target):
            wrapper_class = TLCPUSourceWrapper
        elif is_metal_target(self.target):
            wrapper_class = TLMetalSourceWrapper
        else:
            raise ValueError(f"Unsupported target for source adapter: {self.target}")
        wrapper = wrapper_class(
            scheduled_ir_module=self.scheduled_ir_module,
            source=c_source,
            target=self.target,
            device_mod=self.device_mod,
            host_mod=self.host_mod,
            pass_configs=self.pass_configs,
        )
        return wrapper.lib_code


class TLPyWrapper(TLWrapper):
    def __init__(self, target: Target):
        super().__init__(target)

    def wrap(self, py_source: str):
        # assert self.scheduled_ir_module is not None, "Please assign optimized module first."
        if is_cutedsl_target(self.target):
            from tilelang.jit.adapter.cutedsl import TLCuTeDSLSourceWrapper

            wrapper_class = TLCuTeDSLSourceWrapper
        elif is_cuda_target(self.target):
            from tilelang.jit.adapter.nvrtc import TLNVRTCSourceWrapper

            wrapper_class = TLNVRTCSourceWrapper
        else:
            raise ValueError(f"Unsupported target for NVRTC backend: {self.target}")
        wrapper = wrapper_class(
            scheduled_ir_module=self.scheduled_ir_module,
            source=py_source,
            target=self.target,
            device_mod=self.device_mod,
            host_mod=self.host_mod,
            pass_configs=self.pass_configs,
        )
        return {
            "host_func": getattr(wrapper, "host_func", None),
            "function_names": getattr(wrapper, "function_names", None),
            "tma_cpp_init_code": getattr(wrapper, "tma_cpp_init_code", None),
            "tma_lib_name": getattr(wrapper, "tma_lib_name", None),
            "launcher_cpp_code": getattr(wrapper, "launcher_cpp_code", None),
            "launcher_lib_name": getattr(wrapper, "launcher_lib_name", None),
        }
