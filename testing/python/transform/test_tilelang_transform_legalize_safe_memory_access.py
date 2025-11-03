from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def vectorize_access_legalize(M: int = 64, N: int = 64, M_offset: int = 2, N_offset: int = 2):
    dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype=dtype),):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N), dtype=dtype)
            tid = T.get_thread_binding()
            for j in T.serial(N):
                A_shared[tid, j] = A[tid + M_offset, j + N_offset]

    @T.prim_func
    def expected(A: T.Tensor((M, N), dtype=dtype),):
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


def issue_1013_buggy_kernel():
    # NOTE: This kernel is mainly to test some corner cases in boundary check

    num_tokens = T.dynamic('num_tokens')
    num_threads = 128

    @T.prim_func
    def main(x: T.Tensor((num_tokens,), dtype="int64")):
        with T.Kernel(1, threads=num_threads) as _:
            count = T.alloc_var('int')
            thread_idx = T.get_thread_binding()
            for i in T.serial(0, T.ceildiv(num_tokens - thread_idx, num_threads)):
                idx = thread_idx + i * num_threads
                count += x[idx] == 2

    # NOTE(chaofan): Ideally, the prover should be able to prove that the access is safe
    # and the padding value is not used. However, the current prover cannot handle this case.
    # So for now the expected kernel is a if-else statement to check the boundary.
    @T.prim_func
    def expected(x: T.Tensor((num_tokens,), dtype="int64")):
        with T.Kernel(1, threads=num_threads) as _:
            count = T.alloc_var('int')
            thread_idx = T.get_thread_binding()
            for i in T.serial(0, T.ceildiv(num_tokens - thread_idx, num_threads)):
                idx = thread_idx + i * num_threads
                count += T.Cast("int32",
                                T.if_then_else(idx < num_tokens, x[idx], T.int64(0)) == T.int64(2))

    return main, expected


def vectorize_access_with_atmoic_add_legalize(M: int = 64,
                                              N: int = 64,
                                              M_offset: int = 2,
                                              N_offset: int = 2):
    dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype=dtype),):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N), dtype=dtype)
            tid = T.get_thread_binding()
            for j in T.serial(N):
                A_shared[tid, j] = A[tid + M_offset, j + N_offset]
                T.atomic_add(A[tid + M_offset, j + N_offset], 1)

    @T.prim_func
    def expected(A: T.Tensor((M, N), dtype=dtype),):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N), dtype=dtype)
            tid = T.get_thread_binding()

            T.reads(A[tid + M_offset, N_offset:N + N_offset])
            for j in T.serial(N):
                A_shared[tid, j] = T.if_then_else(
                    j + N_offset < N,
                    T.if_then_else(tid + M_offset < M, A[tid + M_offset, j + N_offset],
                                   T.float32(0)), T.float32(0))
                # Nest if-then-else is expected, do not flatten it to pass structural equal check
                if j + N_offset < N:  # noqa: SIM102
                    if tid + M_offset < M:
                        T.call_extern("handle", "AtomicAdd", A[tid + M_offset, j + N_offset], 1)

    return main, expected


def assert_vectorize_access_with_atmoic_add(M: int = 64, N: int = 64):
    func, expected = vectorize_access_with_atmoic_add_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, expected.body)


def oob_store_legalize(M: int = 64, N: int = 64, M_offset: int = 2, N_offset: int = 2):
    dtype = "float32"

    @T.prim_func
    def main(A: T.Tensor((M, N), dtype=dtype),):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            for j in T.serial(N):
                A[tid + M_offset, j + N_offset] = 1

    @T.prim_func
    def expected(A: T.Tensor((M, N), dtype=dtype),):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            T.writes(A[tid + M_offset, N_offset:N + N_offset])
            for j in T.serial(N):
                if j + N_offset < N:  # noqa: SIM102
                    if tid + M_offset < M:
                        A[tid + M_offset, j + N_offset] = T.float32(1.0)

    return main, expected


def assert_oob_store_legalize(M: int = 64, N: int = 64):
    func, expected = oob_store_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, expected.body)


def test_vectorize_access():
    assert_vectorize_access(64, 64)


def test_issue_1013():
    func, expected = issue_1013_buggy_kernel()
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    transformed = tl.transform.LegalizeSafeMemoryAccess()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, expected.body)


def test_vectorize_access_with_atmoic_add():
    assert_vectorize_access_with_atmoic_add(64, 64)


def test_oob_store():
    assert_oob_store_legalize(64, 64)


if __name__ == "__main__":
    tilelang.testing.main()
