"""PTO tensor storage handling for TVM-FFI adapters."""

from tilelang.jit.adapter.ascend.tvm_ffi import AscendTVMFFIKernelAdapter


class PTOTVMFFIKernelAdapter(AscendTVMFFIKernelAdapter):
    def _get_param_shapes(self):
        return [param.storage_shape(target=self.target) for param in self.params]
