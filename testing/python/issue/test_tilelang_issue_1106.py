import tilelang
import tilelang.testing
from tilelang import language as T


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    },
)
def get_kernel(m: int):
    dtype = "int32"

    @T.prim_func
    def test_kernel(a: T.Tensor[(m,), dtype], b: T.Tensor[(m,), dtype]):
        with T.Kernel(1, threads=64) as (bx):
            shared = T.alloc_shared((64,), dtype)
            tx = T.get_thread_binding(0)
            tid = tx + bx * 64

            for i in T.serial((m // 2 - tx) // 64 + 1):
                for j in T.vectorized(2):
                    shared[tx] += a[(i * 64 + tid) * 2 + j]

            b[tid] = shared[tx]

    return test_kernel


def test_issue_1106():
    m = 200
    kernel = get_kernel(m)
    assert "__syncthreads" not in kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
