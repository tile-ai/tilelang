"""NVRTC compilation with native TIRx host and argument contracts."""

from __future__ import annotations

from tvm_tirx import ir, tirx
from tvm_tirx.target import Target

from tilelang.backend.target import determine_target
from tilelang.jit.adapter.base import BaseKernelAdapter
from tilelang.jit.adapter.device_py_host import NativeDeviceArguments
from tilelang.jit.adapter.wrapper import TLPyWrapper
from tilelang.utils.language import retrieve_func_from_module

from . import check_nvrtc_available
from .libgen import NVRTCLibraryGenerator


class NVRTCKernelAdapter(NativeDeviceArguments, BaseKernelAdapter):
    """Compile cubin with NVRTC; reuse the provider's CUDA runtime to launch it."""

    # NVRTC observes the caller's byte-carrier shapes and allocates outputs
    # with those shapes. Native packed Tensor ABIs use logical sub-byte shapes.
    _source_packed_storage = False

    def _bind_source_arguments(self, plan, arguments):
        return plan.bind_arguments(arguments, nvrtc_storage=True)

    def __init__(
        self,
        params,
        result_idx,
        target,
        func_or_mod,
        host_mod=None,
        device_mod=None,
        device_kernel_source=None,
        verbose=False,
        pass_configs=None,
        compile_flags=None,
    ):
        check_nvrtc_available()
        self.params = params
        self.result_idx = self._legalize_result_idx(result_idx)
        self.ir_module = self._as_module(func_or_mod)
        self._initialize_native_arguments()
        self.device_kernel_source = device_kernel_source
        self.target = Target(determine_target(target))
        self.verbose = verbose
        self.wrapper = TLPyWrapper(self.target)
        self.wrapper.assign_optimized_module(self.ir_module)
        self.wrapper.assign_pass_configs(pass_configs)
        self.wrapper.assign_host_module(host_mod)
        self.wrapper.assign_device_module(device_mod)
        result = self.wrapper.wrap(device_kernel_source)
        self.host_func = result["host_func"]
        self.function_names = result["function_names"]
        self.lib_generator = NVRTCLibraryGenerator(self.target, verbose)
        self.lib_generator.update_lib_code(device_kernel_source)
        self.lib_generator.update_host_func(self.host_func)
        self.lib_generator.assign_compile_flags(compile_flags)
        self.lib_generator.compile_lib()
        self.lib_generator.load_lib()
        self.libpath = self.lib_generator.libpath
        self.pymodule = self.lib_generator.pymodule
        self.kernels = self.lib_generator.kernels
        self._post_init()

    @staticmethod
    def _as_module(function_or_module):
        if isinstance(function_or_module, tirx.Function):
            return ir.IRModule({function_or_module.attrs["global_symbol"]: function_or_module})
        if not isinstance(function_or_module, ir.IRModule):
            raise TypeError("NVRTC requires a native Function or IRModule")
        return function_or_module

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
        check_nvrtc_available()
        adapter = cls.__new__(cls)
        adapter.params = params
        adapter.result_idx = adapter._legalize_result_idx(result_idx)
        adapter.ir_module = cls._as_module(func_or_mod)
        adapter._initialize_native_arguments()
        adapter._set_cached_text_source("host_func", "_host_kernel_source_path", host_kernel_source)
        adapter._set_cached_text_source("device_kernel_source", "_device_kernel_source_path", device_kernel_source)
        adapter.target = Target(determine_target(target))
        adapter.verbose = verbose
        adapter.lib_generator = NVRTCLibraryGenerator(adapter.target, verbose)
        adapter.lib_generator.assign_compile_flags(compile_flags)
        adapter.lib_generator.load_lib(lib_path=kernel_lib_path)
        adapter.libpath = adapter.lib_generator.libpath
        adapter.pymodule = adapter.lib_generator.pymodule
        adapter.kernels = adapter.lib_generator.kernels
        adapter.function_names = adapter.pymodule._function_names
        adapter._post_init()
        return adapter

    def get_kernel_source(self, kernel_only=True):
        if kernel_only:
            return self._load_cached_text_source("device_kernel_source", "_device_kernel_source_path")
        return self.get_host_source()

    def get_host_source(self):
        return self._load_cached_text_source("host_func", "_host_kernel_source_path")

    def _forward_from_prebuild_lib(self, *args, stream=None, device_id=0):
        return self.pymodule.call(self.kernels, *args, stream=stream, device_id=device_id)

    def _wrap_forward_from_prebuild_lib(self, *inputs, stream=None):
        return self._wrap_native_forward(*inputs, stream=stream)

    def _convert_torch_func(self):
        return self._wrap_forward_from_prebuild_lib

    @property
    def prim_func(self):
        return retrieve_func_from_module(self.ir_module)
