"""Keep logical GEMM regions while preparing native opaque SIMD-group storage."""

import tvm_ffi


# Preserve the existing pass-object public interface. The native implementation
# owns the type-changing Bind/use transaction and per-kernel thread scope.
MetalFragmentToSimdgroup = tvm_ffi.get_global_func("tl.transform.MetalFragmentToSimdgroup")()

__all__ = ["MetalFragmentToSimdgroup"]
