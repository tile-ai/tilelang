from __future__ import annotations

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
)
def _wait_scope_semantics_kernel():
    @T.prim_func
    def main(signal: T.Tensor((1,), T.uint32)):
        with T.Kernel(1, threads=1):
            T.wait_eq(signal[0], 1, scope="gpu", semantics="volatile")

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    }
)
def _multimem_signal_dtype_kernel():
    @T.prim_func
    def main(signal32: T.Tensor((1,), T.uint32), signal64: T.Tensor((1,), T.uint64)):
        with T.Kernel(1, threads=1):
            T.multimem_signal(signal32[0], 1)
            T.multimem_signal(signal64[0], 1)

    return main


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_wait_scope_semantics_codegen():
    source = _wait_scope_semantics_kernel().get_kernel_source()
    assert "tl::wait_eq<tl::WaitScope::kGpu, tl::WaitSemantics::kVolatile>" in source


@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_multimem_signal_infers_dtype_codegen():
    source = _multimem_signal_dtype_kernel().get_kernel_source()
    assert "tl::multimem::Signal<uint32_t>::run" in source
    assert "tl::multimem::Signal<uint64_t>::run" in source
