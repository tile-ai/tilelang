from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def vectorize_access_legalize(M: int = 64, N: int = 64):
    dtype = T.float32
    vec_len = 8

    @T.prim_func
    def main(
        A: T.Tensor((M, N, vec_len), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N, vec_len), dtype=dtype)
            tid = T.get_thread_binding()
            for j in T.serial(N):
                for v in T.vectorized(vec_len):
                    A_shared[tid, j, v] = A[tid, j, v]

    @T.prim_func
    def expected(
        A: T.Tensor((M, N, vec_len), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N, vec_len), dtype=dtype)
            tid = T.get_thread_binding()
            for j, v_2 in T.grid(M, vec_len // 4):
                for vec in T.vectorized(4):
                    A_shared[tid, j, v_2 * 4 + vec] = A[tid, j, v_2 * 4 + vec]

    return main, expected


def assert_vectorize_access(M: int = 64, N: int = 64):
    func, expected = vectorize_access_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    with tvm.target.Target("cuda"):
        transformed = tl.transform.LegalizeVectorizedLoop()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, expected.body)


def test_vectorize_access():
    assert_vectorize_access(64, 64)


def test_vectorized_nonzero_min_uses_aligned_vector_width():
    @T.prim_func
    def before(
        A: T.Tensor((8,), T.int32),
        B: T.Tensor((8,), T.int32),
    ):
        for value in T.vectorized(2, 6):
            B[value] = A[value] + 1

    @T.prim_func
    def expected(
        A: T.Tensor((8,), T.int32),
        B: T.Tensor((8,), T.int32),
    ):
        for value_outer in T.serial(2):
            for vec in T.vectorized(2):
                B[2 + (value_outer * 2 + vec)] = A[2 + (value_outer * 2 + vec)] + 1

    mod = tvm.IRModule.from_expr(before)
    with tvm.target.Target("cuda"):
        transformed = tl.transform.LegalizeVectorizedLoop()(mod)
    tvm.ir.assert_structural_equal(transformed["before"].body, expected.body)


if __name__ == "__main__":
    tilelang.testing.main()
