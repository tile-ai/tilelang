import tilelang
import tilelang.testing
from tilelang import language as T


def _compile_cuda(func):
    tilelang.disable_cache()
    try:
        return tilelang.compile(func, target="cuda", execution_backend="tvm_ffi")
    finally:
        tilelang.enable_cache()


@tilelang.testing.requires_cuda
def test_sync_threads_with_variable_barrier_id():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=256) as (bx,):
            barrier_id = T.int32(1)
            T.sync_threads(barrier_id)
            T.sync_threads(barrier_id, 128)

    kernel = _compile_cuda(kernel)
    src = kernel.get_kernel_source()
    assert "tl::__sync_thread_partial(" in src, src


@tilelang.testing.requires_cuda
def test_sync_threads_with_computed_barrier_id():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=256) as (bx,):
            tx = T.launch_thread("threadIdx.x", 256)
            barrier_id = tx % 4
            T.sync_threads(barrier_id, 256)

    kernel = _compile_cuda(kernel)
    src = kernel.get_kernel_source()
    assert "tl::__sync_thread_partial(" in src, src


if __name__ == "__main__":
    tilelang.testing.main()
