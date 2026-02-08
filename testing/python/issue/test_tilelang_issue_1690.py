import tilelang
import tilelang.testing
import tilelang.language as T


def test_issue_1690():
    @tilelang.jit()
    def test(A):
        N = T.const("N")
        A: T.Tensor[[N], T.float32]
        with T.Kernel():
            tmp = T.alloc_fragment((N,), T.float32)
            tmp_max = T.alloc_fragment(1, T.float32)
            T.copy(A, tmp)
            T.reduce_max(tmp, tmp_max, dim=0)

    test.compile(N=16)


if __name__ == "__main__":
    tilelang.testing.main()
