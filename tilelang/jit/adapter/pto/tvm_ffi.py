"""PTO tensor storage handling for TVM-FFI adapters."""

from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter


class PTOTVMFFIKernelAdapter(TVMFFIKernelAdapter):
    def _get_param_shapes(self):
        return [param.storage_shape(target=self.target) for param in self.params]
