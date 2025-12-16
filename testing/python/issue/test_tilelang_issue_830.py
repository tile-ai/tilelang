# ruff: noqa

import torch
import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def _empty_kernel():
    @T.prim_func
    def empty_kernel():
        with T.Kernel(1, threads=32) as thread_idx:
            pass

    return empty_kernel


@tilelang.testing.requires_cuda
def test_empty_kernel_lowering():
    # Ensure a valid CUDA runtime context is current on this thread for the
    # target device before using driver API calls. Without this, calls like
    # cuModuleLoadData can fail with CUDA_ERROR_INVALID_CONTEXT, especially
    # for kernels that don't touch any device memory or streams beforehand
    # (e.g., "empty" kernels) and therefore haven't triggered context
    # creation implicitly.
    torch.cuda.set_device(0)
    kernel = _empty_kernel()
    kernel()


@tilelang.jit
def _empty_with_dead_code_kernel():
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def buggy_kernel(x: T.Tensor[(num_tokens,), T.float32]):
        with T.Kernel(num_tokens, threads=32) as pid:
            y = x[pid]

    return buggy_kernel


@tilelang.testing.requires_cuda
def test_empty_with_dead_code_kernel():
    kernel = _empty_with_dead_code_kernel()
    x = torch.randn((128,), dtype=torch.float32, device="cuda")
    kernel(x)


@tilelang.jit
def _empty_kernel_with_binding_variants(use_tuple_binding: bool = False):
    @T.prim_func
    def kernel_with_tuple_kernel_binding():
        with T.Kernel(1, threads=32) as (pid,):
            print(pid)
            pass

    @T.prim_func
    def kernel_with_scalar_kernel_binding():
        with T.Kernel(1, threads=32) as pid:
            print(pid)
            pass

    return kernel_with_tuple_kernel_binding if use_tuple_binding else kernel_with_scalar_kernel_binding


@tilelang.testing.requires_cuda
def test_empty_kernel_with_binding_variants():
    torch.cuda.set_device(0)
    kernel = _empty_kernel_with_binding_variants()
    kernel()

    tuple_kernel = _empty_kernel_with_binding_variants(use_tuple_binding=True)
    tuple_kernel()


if __name__ == "__main__":
    tilelang.testing.main()
