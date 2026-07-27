# ruff: noqa
import tilelang
import tilelang.testing
from tilelang import language as T


@tilelang.testing.requires_cuda
def test_issue_1257_missing_syncthreads_after_atomic_add_on_shared():
    """Regression for issue #1257.

    After an AtomicAdd on shared memory, a __syncthreads() barrier is required
    before other threads read from shared memory. Without the barrier, threads
    may read stale values.
    """

    m = 1024

    @tilelang.jit
    def get_kernel(m: int):
        @T.prim_func
        def test_kernel(
            a: T.Tensor[(m,), "int32"],
        ):
            with T.Kernel(1, threads=1024) as (bx):
                shared = T.alloc_shared((1024,), "int32")
                tx = T.get_thread_binding(0)
                shared[tx ^ 1] = 0
                T.atomic_add(shared[tx], 1)
                a[tx] = shared[tx ^ 32]

        return test_kernel

    kernel = get_kernel(m)
    source = kernel.get_kernel_source()

    sync_threads = source.count("__syncthreads()")
    assert sync_threads == 2, "Missing __syncthreads() between AtomicAdd on shared memory and subsequent shared memory read"


if __name__ == "__main__":
    tilelang.testing.main()
