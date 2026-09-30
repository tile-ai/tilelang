"""Cython execution adapter for AscendC kernels."""

from tilelang.jit.adapter.cython import CythonKernelAdapter

from .libgen import AscendLibraryGenerator
from .wrapper import TLAscendWrapper


class AscendCythonKernelAdapter(CythonKernelAdapter):
    wrapper_class = TLAscendWrapper
    library_generator_class = AscendLibraryGenerator
