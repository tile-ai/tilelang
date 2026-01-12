import tilelang
import tilelang.testing
from tilelang import tvm as tvm
from tilelang import language as T


def test_unroll_with_step():
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype=T.float32, align=16)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for _threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                for i in T.unroll(0, 16, step=4):
                    A[0, i] = 1.0

    kernel = tilelang.compile(main)
    assert "#pragma unroll" in kernel.get_kernel_source()


# TODO: unroll factor is not supported on hip, skip.
@tilelang.testing.requires_cuda
def test_unroll_with_unroll_factor():
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype=T.float32, align=16)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for _threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                for i in T.unroll(0, 16, unroll_factor=4):
                    A[0, i] = 1.0

    kernel = tilelang.compile(main)
    assert "#pragma unroll 4" in kernel.get_kernel_source()


def test_unroll_with_extent_only():
    """Test T.unroll with only extent parameter."""

    @tilelang.jit
    def unroll_kernel():
        out = T.empty((512,), dtype=T.float32)
        with T.Kernel(1, threads=512):
            tid = T.get_thread_binding()
            for i in T.unroll(tid % 32):
                out[i] = i
        return out

    kernel = unroll_kernel.compile()
    source = kernel.get_kernel_source()
    assert "#pragma unroll" in source


if __name__ == "__main__":
    tilelang.testing.main()
