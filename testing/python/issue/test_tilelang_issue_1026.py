import tilelang
import tilelang.testing
from tilelang import language as T


@tilelang.jit
def get_shared_kernel():
    @T.prim_func
    def shared_kernel():
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared_mem = T.alloc_shared((32), dtype="float32", scope="shared")
            if tx % 2 == 0:
                a = shared_mem[tx]
                shared_mem[tx ^ 1] = a

    return shared_kernel


def test_issue_1026():
    kernel = get_shared_kernel()
    assert "__syncthreads" not in kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
