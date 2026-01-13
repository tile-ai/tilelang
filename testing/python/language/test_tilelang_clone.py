import torch
import tilelang as tl
import tilelang.language as T


def test_clone_buffer():
    for scope in ["shared", "local"]:
        M, N = 64, 64
        @tl.jit
        def kernel(A: T.Tensor((M, N), "float16"), B: T.Tensor((M, N), "float16")):
            with T.Kernel(1, 1):
                buf_source = T.alloc_buffer((M, N), dtype="float16", scope=scope)
                T.copy(A, buf_source)
                buf_cloned = T.clone_buffer(buf_source)
                T.copy(buf_cloned, B)

        a = torch.randn(M, N).cuda().half()
        b = torch.empty(M, N).cuda().half()
        kernel(a, b)
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-3)
        print("Test clone_buffer: Passed")


def test_load_buffer():
    K = 32
    @tl.jit
    def kernel(A: T.Tensor((K, K), "float16"), B: T.Tensor((K, K), "float16")):
        with T.Kernel(1, 1, threads=K * K):
            A_shared = T.load_buffer(A, offsets=(0, 0), shape=(K, K), target="shared")
            for i, j in T.Parallel(K, K):
                A_shared[i, j] = A[i, j]
            T.sync_threads()
            for i, j in T.Parallel(K, K):
                B[i, j] = A_shared[i, j]

    a = torch.randn(K, K).cuda().half()
    b = torch.zeros(K, K).cuda().half()
    kernel(a, b)
    torch.testing.assert_close(b, a, rtol=1e-3, atol=1e-3)
    print("Test load_buffer: Passed ")


if __name__ == "__main__":
    test_clone_buffer()
    test_load_buffer()
