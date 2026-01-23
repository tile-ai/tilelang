import tilelang
import tilelang.testing
import tilelang.language as T


def test_tilelang_issue_1719():
    @tilelang.jit()
    def _buggy_kernel(M: int, N: int) -> tilelang.JITKernel:
        @T.prim_func
        def kernel() -> None:
            with T.Kernel():
                tmp1 = T.alloc_fragment((N, M), T.float32)
                tmp2 = T.alloc_fragment((N, M), T.float32)
                tmp3 = T.alloc_fragment((N, M, M), T.float32)
                for i, j, k in T.Parallel(N, M, M):
                    tmp3[i, j, k] = 1
                T.reduce_sum(tmp3, tmp2, dim=1)
                for i, k in T.Parallel(N, M):
                    tmp2[i, k] /= tmp1[i, k]

        return kernel

    kernel = _buggy_kernel(M=4, N=32)
    assert "tmp2[(((int)threadIdx.x) & 3)]" not in kernel.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
