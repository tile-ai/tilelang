import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit
def dynamic_shared_scan(rank=1, dynamic_axis=0, dim=0, reverse=False):
    n = T.dynamic("n")
    shape = (n,) if rank == 1 else ((n, 16) if dynamic_axis == 0 else (16, n))

    @T.prim_func
    def kernel(A: T.Tensor(shape, "int32"), B: T.Tensor(shape, "int32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared(shape, "int32")
            # Isolate shape inference from TMA transfer-alignment constraints.
            T.copy(A, shared, disable_tma=True)
            T.cumsum(shared, shared, dim=dim, reverse=reverse)
            T.copy(shared, B, disable_tma=True)

    return kernel


@pytest.mark.parametrize("rank,dynamic_axis,dim", [(1, 0, 0), (2, 0, 0), (2, 0, 1), (2, 1, 0), (2, 1, 1)])
@pytest.mark.parametrize("reverse", [False, True])
@tilelang.testing.requires_cuda
def test_dynamic_shared_scan(rank, dynamic_axis, dim, reverse):
    kernel = dynamic_shared_scan(rank, dynamic_axis, dim, reverse)
    for n in [1, 7, 31, 32, 33, 127, 128, 129, 257]:
        shape = (n,) if rank == 1 else ((n, 16) if dynamic_axis == 0 else (16, n))
        a = torch.randint(-10, 11, shape, device="cuda", dtype=torch.int32)
        b = torch.empty_like(a)
        kernel(a, b)
        expected = a.flip([dim]) if reverse else a
        expected = expected.cumsum(dim, dtype=torch.int32)
        if reverse:
            expected = expected.flip([dim])
        torch.testing.assert_close(b, expected, rtol=0, atol=0)


@tilelang.jit
def dynamic_transpose_scan():
    n = T.dynamic("n")

    @T.prim_func
    def kernel(A: T.Tensor((n, 16), "int32"), B: T.Tensor((n, 16), "int32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, n), "int32")
            for d, b in T.Parallel(16, n):
                shared[d, b] = A[b, d]
            T.cumsum(shared, shared, dim=1)
            for d, b in T.Parallel(16, n):
                B[b, d] = shared[d, b]

    return kernel


@tilelang.testing.requires_cuda
def test_dynamic_transpose_scan_graph():
    kernel = dynamic_transpose_scan()
    for n in [7, 129, 257]:
        a = torch.randint(-10, 11, (n, 16), device="cuda", dtype=torch.int32)
        b = torch.empty_like(a)
        kernel(a, b)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel(a, b)
        for _ in range(3):
            a.random_(-10, 11)
            graph.replay()
            torch.testing.assert_close(b, a.cumsum(0, dtype=torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("mapping", ["drop_row", "drop_column", "overlap", "wrap_row"])
@tilelang.testing.requires_cuda
def test_symbolic_noninjective_shared_layout_rejected(mapping):
    from tilelang.layout import Layout

    n = T.dynamic("n")
    if mapping == "drop_row":
        layout = Layout((16, n), lambda i, j: j)
    elif mapping == "drop_column":
        layout = Layout((16, n), lambda i, j: i)
    elif mapping == "overlap":
        # Both coordinates occur, but adjacent rows overlap at their ends.
        layout = Layout((16, n), lambda i, j: i * (n - 1) + j)
    else:
        # Rows i and i + 8 collide despite both coordinates being present.
        layout = Layout((16, n), lambda i, j: (i % 8) * n + j)

    @T.prim_func
    def kernel(A: T.Tensor((16, n), "int32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, n), "int32")
            T.annotate_layout({shared: layout})
            for i, j in T.Parallel(16, n):
                shared[i, j] = A[i, j]
            for i, j in T.Parallel(16, n):
                A[i, j] = shared[i, j]

    with pytest.raises(ValueError, match="must be injective"):
        tilelang.compile(kernel)
