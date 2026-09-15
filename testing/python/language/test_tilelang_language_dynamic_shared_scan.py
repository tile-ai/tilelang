import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import Layout


@tilelang.testing.requires_cuda
def test_dynamic_shared_scan():
    n = T.dynamic("n")

    @T.prim_func
    def kernel(A: T.Tensor((16, n), "int32"), B: T.Tensor((16, n), "int32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, n), "int32")
            T.copy(A, shared, disable_tma=True)
            T.cumsum(shared, shared, dim=1)
            T.copy(shared, B, disable_tma=True)

    compiled = tilelang.compile(kernel, out_idx=[1])
    for length in (7, 129):
        a = torch.randint(-10, 11, (16, length), device="cuda", dtype=torch.int32)
        torch.testing.assert_close(compiled(a), a.cumsum(1, dtype=torch.int32), rtol=0, atol=0)


@tilelang.testing.requires_cuda
def test_symbolic_noninjective_shared_layout_rejected():
    n = T.dynamic("n")

    @T.prim_func
    def kernel(A: T.Tensor((16, n), "int32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, n), "int32")
            # Both coordinates occur, but adjacent rows overlap.
            T.annotate_layout({shared: Layout((16, n), lambda i, j: i * (n - 1) + j)})
            for i, j in T.Parallel(16, n):
                shared[i, j] = A[i, j]
            for i, j in T.Parallel(16, n):
                A[i, j] = shared[i, j]

    with pytest.raises(ValueError, match="must be injective"):
        tilelang.compile(kernel)


if __name__ == "__main__":
    tilelang.testing.main()
