import tilelang as tl
import tilelang.testing
import tilelang.language as T


def test_issue_1374_non_var_itermark():
    @tl.jit
    def get_wrong_kernel(M: int = 4096):
        dtype = "int32"
        num_threads = 128

        @T.prim_func
        def main(A: T.Tensor((16, 14), dtype=dtype), B: T.Tensor((16, 448), dtype=dtype)):
            with T.Kernel(1, threads=num_threads) as (bx,):
                A_local = T.alloc_fragment((16, 14), dtype=dtype)
                B_local = T.alloc_fragment((16, 448), dtype=dtype)

                T.copy(A, A_local)
                T.copy(B, B_local)
                for i, j in T.Parallel(16, 448):
                    A_local[i, j // 32] += B[i, j]

        return main

    kernel = get_wrong_kernel()
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    tilelang.testing.main()
