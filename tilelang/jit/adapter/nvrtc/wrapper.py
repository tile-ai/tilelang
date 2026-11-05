"""NVRTC host generation from the retained native Function graph."""

from tilelang.jit.adapter.device_py_host import NativePythonHost
from tilelang.jit.adapter.wrapper import _require_lowered_modules
from tilelang.utils.language import retrieve_func_from_module


class TLNVRTCSourceWrapper:
    """Keep CUDA device source and emit its real host statement/call graph."""

    def __init__(self, scheduled_ir_module, source, target, device_mod=None, host_mod=None, pass_configs=None):
        self.mod = scheduled_ir_module
        self.source = source
        self.target = target
        self.pass_configs = pass_configs
        self.device_mod, self.host_mod = _require_lowered_modules(device_mod, host_mod)
        self.prim_func = retrieve_func_from_module(scheduled_ir_module)
        self.native_host = NativePythonHost(self.prim_func, self.host_mod, self.device_mod)
        self.host_func = self.native_host.emit()
        self.function_names = list(self.native_host.function_names)
        self.lib_code = source

    def update_lib_code(self, code):
        self.source = self.lib_code = code
        return code
