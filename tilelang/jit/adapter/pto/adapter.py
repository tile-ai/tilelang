"""Cython execution adapter for PTODSL kernels."""

from tilelang.jit.adapter.cython import CythonKernelAdapter

from .libgen import PTOLibraryGenerator
from .wrapper import TLPTOWrapper


class PTOCythonKernelAdapter(CythonKernelAdapter):
    wrapper_class = TLPTOWrapper
    library_generator_class = PTOLibraryGenerator

    def _compile_library(self):
        wrapper = self.wrapper.source_wrapper
        self.lib_generator.update_pto_kernels(wrapper.pto_kernel_source, wrapper.pto_kernel_names)
        super()._compile_library()
