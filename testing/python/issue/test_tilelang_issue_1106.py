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
    source = kernel.get_kernel_source()
    # Ensure __syncthreads is not inside the for loop
    for_start = source.find("for (int i = 0;")
    for_end = source.find("__syncthreads")
    assert for_end > for_start, "__syncthreads should be after the for loop, not inside it"
    # Check that __syncthreads appears after the closing brace of the outer for loop
    assert source[for_end - 4 : for_end - 2] == "}\n", "__syncthreads should not be inside any for loop"


if __name__ == "__main__":
    # tilelang.testing.main()
    test_issue_1106()
