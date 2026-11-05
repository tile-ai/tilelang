"""CuTeDSL runtime adapter with native tensor, shape and host contracts."""

from __future__ import annotations

from numbers import Integral
from operator import index as integer_index
from pathlib import Path

from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang.backend.target import determine_target
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.jit.adapter.device_py_host import NativeDeviceArguments
from tilelang.jit.adapter.wrapper import TLPyWrapper
from tilelang.utils.language import retrieve_func_from_module

from .checks import check_cutedsl_available
from .libgen import CuTeDSLLibraryGenerator


class CuTeDSLKernelAdapter(NativeDeviceArguments, BaseKernelAdapter):
    """Keep the public call convention while lowering to one native host ABI."""

    def __init__(
        self,
        params,
        result_idx,
        target,
        func_or_mod,
        host_mod=None,
        device_mod=None,
        host_kernel_source=None,
        device_kernel_source=None,
        verbose=False,
        pass_configs=None,
        compile_flags=None,
    ):
        check_cutedsl_available()
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.ir_module = self._as_module(func_or_mod)
        self._initialize_native_arguments()
        self.target = Target(determine_target(target))
        self.verbose = verbose
        self.device_kernel_source = self.kernel_global_source = device_kernel_source
        self.wrapper = TLPyWrapper(self.target)
        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        result = self.wrapper.wrap(device_kernel_source)
        self.host_func = self.host_kernel_source = result["host_func"]
        self.function_names = result["function_names"]
        self.launcher_cpp_code = self.launcher_lib_name = None
        self.lib_generator = CuTeDSLLibraryGenerator(self.target, verbose)
        self.lib_generator.update_lib_code(device_kernel_source)
        self.lib_generator.update_host_func(self.host_func)
        self.lib_generator.assign_compile_flags(compile_flags)
        self.lib_generator.compile_lib()
        self.lib_generator.load_lib()
        self._finish_init()

    @staticmethod
    def _as_module(function_or_module):
        if isinstance(function_or_module, tirx.Function):
            symbol = function_or_module.attrs.get("global_symbol") if function_or_module.attrs else None
            if symbol is None:
                raise ValueError("Function is missing required attr 'global_symbol'")
            return ir.IRModule({symbol: function_or_module})
        if not isinstance(function_or_module, ir.IRModule):
            raise TypeError("CuTeDSL requires a native Function or IRModule")
        return function_or_module

    def _finish_init(self):
        self.libpath = self.lib_generator.libpath
        self.pymodule = self.lib_generator.pymodule
        self.kernels = self.lib_generator.runtime.kernels
        # The loaded Python module owns execution. Its source is optional
        # diagnostic material, and may disappear after a successful import.
        # A missing/unreadable module still fails inside load_lib before this.
        try:
            self.generated_module_source = Path(self.libpath).read_text()
        except OSError:
            self.generated_module_source = None
        self._post_init()

    @classmethod
    def from_database(
        cls,
        params,
        result_idx,
        target,
        func_or_mod,
        host_kernel_source,
        device_kernel_source,
        kernel_lib_path,
        verbose=False,
        pass_configs=None,
        compile_flags=None,
    ):
        # Cached provider launches need no CUTLASS import or compiler. A cache
        # miss and the optional direct launcher perform the capability check.
        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter.ir_module = cls._as_module(func_or_mod)
        adapter._initialize_native_arguments()
        adapter._set_cached_text_source("host_kernel_source", "_host_kernel_source_path", host_kernel_source)
        adapter._set_cached_text_source("device_kernel_source", "_device_kernel_source_path", device_kernel_source)
        from tilelang.cuda.target import normalize_cutedsl_target

        adapter.target = normalize_cutedsl_target(target) or Target(determine_target(target))
        adapter.verbose = verbose
        adapter.launcher_cpp_code = adapter.launcher_lib_name = None
        adapter.lib_generator = CuTeDSLLibraryGenerator(adapter.target, verbose)
        adapter.lib_generator.assign_compile_flags(compile_flags)
        adapter.lib_generator.load_lib(lib_path=kernel_lib_path)
        adapter.function_names = list(adapter.lib_generator.pymodule._function_names)
        adapter._finish_init()
        return adapter

    def get_host_source(self):
        return self._load_cached_text_source("host_kernel_source", "_host_kernel_source_path")

    def get_generated_module_source(self):
        return self.generated_module_source

    def get_kernel_source(self, kernel_only=True):
        device_source = self._load_cached_text_source("device_kernel_source", "_device_kernel_source_path")
        if kernel_only:
            return device_source
        return "\n\n".join(source for source in (device_source, self.get_host_source()) if source)

    def _bind_source_arguments(self, plan, arguments):
        # The original CuTe launcher accepted integer carriers for Boolean
        # parameters, then truncated to i1 inside the device kernel. Preserve
        # that boundary before the common Python host applies bool(value).
        # NVRTC and Cython intentionally keep their own truthiness policies.
        for index, parameter in enumerate(plan.function.params):
            if parameter.ty == ir.PrimType("bool"):
                if not isinstance(arguments[index], Integral):
                    raise TypeError(f"CuTeDSL boolean scalar parameter {index} requires an integer")
                value = integer_index(arguments[index])
                if not -(1 << 63) <= value < 1 << 63:
                    raise OverflowError(f"CuTeDSL boolean scalar parameter {index} exceeds the int64 carrier")
                arguments[index] = bool(value & 1)
        return super()._bind_source_arguments(plan, arguments)

    def _forward_from_prebuild_lib(self, *args, stream=None, device_id=0):
        result = self.pymodule.call(self.kernels, *args, stream=stream, device_id=device_id)
        self._save_cubin_to_cache_if_needed()
        return result

    def _save_cubin_to_cache_if_needed(self):
        cache_path = getattr(self, "_cache_path", None)
        if cache_path is not None:
            self.lib_generator.runtime.export_artifacts(cache_path)

    def _wrap_forward_from_prebuild_lib(self, *inputs, stream=None):
        return self._wrap_native_forward(*inputs, stream=stream)

    def _convert_torch_func(self):
        return self._wrap_forward_from_prebuild_lib

    def cleanup(self):
        self.lib_generator.cleanup()

    @property
    def prim_func(self):
        return retrieve_func_from_module(self.ir_module)
