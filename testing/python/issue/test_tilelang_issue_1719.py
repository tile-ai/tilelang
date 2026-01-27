import tilelang
import torch
import tilelang.testing
import tilelang.language as T


@tilelang.testing.requires_cuda
def test_issue_1719_layout_1():
    @tilelang.jit
    def _buggy_kernel():
        with T.Kernel(threads=32):
            tmp1 = T.alloc_shared([32, 32], T.float16)
            tmp2 = T.alloc_shared([32, 32], T.float16)
            tmp3 = T.alloc_fragment([32, 32], T.float32)
            tmp4 = T.alloc_fragment([32], T.float32)
            T.gemm(tmp1, tmp2, tmp3, transpose_B=True)
            T.reduce_max(tmp3, tmp4)
            for i in T.Parallel(32):
                tmp4[i] = 1

    kernel = _buggy_kernel.compile()
    print(kernel.get_kernel_source())


def test_issue_1719_layout_2():
    @tilelang.jit
    def _buggy_kernel(M: int, N: int):
        with T.Kernel():
            tmp1 = T.alloc_fragment((N, M), T.float32)
            tmp2 = T.alloc_fragment((N, M), T.float32)
            tmp3 = T.alloc_fragment((N, M, M), T.float32)
            for i, j, k in T.Parallel(N, M, M):
                tmp3[i, j, k] = 1
            T.reduce_sum(tmp3, tmp2, dim=1)
            for i, k in T.Parallel(N, M):
                tmp2[i, k] /= tmp1[i, k]

    kernel = _buggy_kernel.compile(M=4, N=32)
    print(kernel.get_kernel_source())
    assert "tmp2[(((int)threadIdx.x) & 3)]" not in kernel.get_kernel_source()


@tilelang.testing.requires_cuda
def test_issue_1719_layout_3():
    @tilelang.jit
    def _buggy_kernel(A, dtype=T.float32):
        M, N = T.const("M, N")
        A: T.Tensor[(M, N), dtype]
        B = T.empty((M,), dtype)
        with T.Kernel(1, threads=32) as _:
            A_local = T.alloc_fragment((M, N), dtype)
            B_local = T.alloc_fragment((M,), dtype)

            T.copy(A, A_local)
            T.reduce_sum(A_local, B_local, dim=1)
            T.copy(B_local, B)
        return B

    M, N = 2, 128
    kernel = _buggy_kernel.compile(M=M, N=N)
    a = torch.randn(M, N, device="cuda")
    b = kernel(a)
    print(b, a.sum(dim=1))
    torch.testing.assert_close(b, a.sum(dim=1), atol=1e-2, rtol=1e-2)


def test_issue_1719_layout_4():
    @tilelang.jit
    def _buggy_kernel():
        with T.Kernel(threads=128):
            Q_tail_shared = T.alloc_shared([32, 32], T.bfloat16)
            K_tail_shared = T.alloc_shared([32, 32], T.bfloat16)
            acc_s = T.alloc_fragment([32, 32], T.float32)
            m_i = T.alloc_fragment([32], T.float32)
            T.gemm(Q_tail_shared, K_tail_shared, acc_s, transpose_B=True)
            T.reduce_max(acc_s, m_i)

    _buggy_kernel.compile()


def test_issue_1719_layout_5():
    @tilelang.jit
    def buggy_kernel(A, dtype=T.float32):
        N = T.const("N")
        A: T.Tensor[(1, N), dtype]
        with T.Kernel(1, threads=32) as _:
            A_local = T.alloc_fragment((1, N), dtype)
            B_local = T.alloc_fragment((1,), dtype)

            T.copy(A, A_local)
            T.reduce_sum(A_local, B_local, dim=1)

    buggy_kernel.compile(N=128)


def test_issue_1719_layout_6():
    @tilelang.jit
    def buggy_kernel():
        with T.Kernel():
            tmp1 = T.alloc_fragment((1,), dtype=T.float32)
            tmp2 = T.alloc_fragment((1,), dtype=T.float32)
            tmp1[0] = 1
            T.reduce_sum(tmp1, tmp2)
            tmp2[0]

    buggy_kernel.compile()


def test_issue_1719_layout_7():
    @tilelang.jit
    def buggy_kernel():
        with T.Kernel(threads=32):
            tmp1 = T.alloc_fragment([1, 32], T.float16)
            tmp2 = T.alloc_fragment([32], T.float32)
            tmp3 = T.alloc_fragment([32], T.float32)
            tmp4 = T.alloc_fragment([32], T.float32)
            T.reduce_max(tmp1, tmp4, dim=0)
            k = 0
            T.copy(tmp1[k, :], tmp2)
            for i in T.Parallel(32):
                tmp3[i] += tmp2[i] - tmp4[i]

    buggy_kernel.compile()


if __name__ == "__main__":
    tilelang.testing.main()
