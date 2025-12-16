import tilelang
import tilelang.language as T
import tilelang.testing


def test_assume_remove_boundary_check():
    @tilelang.jit
    def kernel_with_assume():
        N = T.dynamic("N")

        @T.prim_func
        def main(A: T.Tensor((N,), T.float32), l: T.int32, r: T.int32):
            with T.Kernel(1, threads=32) as _:
                for i in T.serial(r - l + 1):
                    T.assume(l + i >= 0 and l + i < N)
                    A[l + i] = 0

        return main

    jit_kernel = kernel_with_assume()
    source = jit_kernel.get_kernel_source()

    assert "if (" not in source


def test_assume_enable_vectorization():
    @tilelang.jit
    def kernel_vectorize(M):
        N = T.dynamic("N")
        vectorize_size = 4

        @T.prim_func
        def main(
            A: T.Tensor((M, N), T.float32),
            B: T.Tensor((M, N), T.float32),
        ):
            with T.Kernel(1, threads=32) as _:
                tid = T.get_thread_binding()

                base_idx = tid * 4
                T.assume(N % vectorize_size == 0)

                for i in T.vectorized(vectorize_size):
                    T.assume(base_idx + i < N)
                    B[tid, base_idx + i] = A[tid, base_idx + i]

        return main

    jit_kernel = kernel_vectorize(128)
    source = jit_kernel.get_kernel_source()

    assert ("float4" in source) and ("if (" not in source)


def test_assume_complex_indexing():
    @tilelang.jit
    def kernel_complex():
        M = T.dynamic("M")
        N = T.dynamic("N")

        @T.prim_func
        def main(
            A: T.Tensor((M, N), T.float32),
            B: T.Tensor((M, N), T.float32),
        ):
            with T.Kernel(1, threads=32) as _:
                tid = T.get_thread_binding()
                for j in T.serial(N):
                    i_src = T.min(j + 233, tid + 2)
                    j_src = j * T.ceildiv(j, i_src) * j - 1

                    T.assume(i_src >= 0 and i_src < M)
                    T.assume(j_src >= 0 and j_src < N)

                    B[tid, j] = A[i_src, j_src]

        return main

    jit_kernel = kernel_complex()
    source = jit_kernel.get_kernel_source()

    assert "if (" not in source


if __name__ == "__main__":
    tilelang.testing.main()
