# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def vectorize_access_legalize(M: int = 64, N: int = 64, M_offset: int = 2, N_offset: int = 2):
    dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype="float32"),):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N), dtype=dtype)
            tid = T.get_thread_binding()
            for j in T.serial(N):
                A_shared[tid, j] = A[tid + M_offset, j + N_offset]

    @T.prim_func
    def expected(A: T.Tensor((M, N), dtype="float32"),):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N), dtype=dtype)
            tid = T.get_thread_binding()

            T.reads(A[tid + M_offset, N_offset:N + N_offset])
            for j in T.serial(N):
                A_shared[tid, j] = T.if_then_else(
                    j + N_offset < N,
                    T.if_then_else(tid + M_offset < M, A[tid + M_offset, j + N_offset],
                                   T.float32(0)), T.float32(0))

    return main, expected


def assert_vectorize_access(M: int = 64, N: int = 64):
    func, expected = vectorize_access_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, expected.body)


def test_vectorize_access():
    assert_vectorize_access(64, 64)


if __name__ == "__main__":
    tilelang.testing.main()
