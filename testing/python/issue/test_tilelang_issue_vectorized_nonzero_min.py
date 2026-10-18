import torch
import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit
def _nonzero_min_vectorized(
    A: T.Tensor((8,), T.int32),
    B: T.Tensor((8,), T.int32),
):
    with T.Kernel(1, threads=1):
        for value in T.vectorized(2, 6):
            B[value] = A[value] + 1


@tilelang.testing.requires_cuda
def test_vectorized_nonzero_min():
    a = torch.arange(8, device="cuda", dtype=torch.int32)
    b = torch.full((8,), -1, device="cuda", dtype=torch.int32)

    _nonzero_min_vectorized(a, b)

    expected = torch.tensor(
        [-1, -1, 3, 4, 5, 6, -1, -1],
        device="cuda",
        dtype=torch.int32,
    )
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("start,stop,step", [(2, 6, 1), (2, 8, 2)])
def test_nested_vectorized_inner_domain_runtime(start, stop, step):
    @T.prim_func
    def main(A: T.Tensor((16,), T.int32), B: T.Tensor((16,), T.int32)):
        with T.Kernel(1, threads=1):
            for j in T.vectorized(2):
                for i in T.serial(start, stop):
                    B[j * 8 + i] = A[j * 8 + i] + 1

    def set_inner_step(node):
        if isinstance(node, tilelang.tvm.tirx.For) and node.kind == tilelang.tvm.tirx.ForKind.SERIAL:
            return tilelang.tvm.tirx.For(node.loop_var, node.min, node.extent, node.kind, node.body, step=step)
        return node

    main = main.with_body(tilelang.tvm.tirx.stmt_functor.ir_transform(main.body, None, set_inner_step, ["tirx.For"]))
    kernel = tilelang.compile(main, target="cuda")
    a = torch.arange(16, device="cuda", dtype=torch.int32)
    b = torch.full_like(a, -1)
    kernel(a, b)
    expected = torch.full_like(a, -1)
    expected.view(2, 8)[:, start:stop:step] = a.view(2, 8)[:, start:stop:step] + 1
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


def _check_mixed_cast_nonzero_min(target, device):
    @T.prim_func
    def main(
        A: T.Tensor((8,), T.float16),
        B: T.Tensor((8,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            a_local = T.alloc_local((8,), T.float32)
            for value in T.vectorized(2, 6):
                if value < 5:
                    a_local[value] = A[value]
            for value in T.vectorized(2, 6):
                if value < 5:
                    B[value] = a_local[value] + 1

    kernel = tilelang.compile(main, target=target, execution_backend="cython")
    # Check that both cast directions actually went through staging.
    source = kernel.get_kernel_source()
    assert "A_local_cast" in source
    assert "B_local_cast" in source

    a = torch.arange(8, device=device, dtype=torch.float16)
    b = torch.full((8,), -1, device=device, dtype=torch.float16)
    kernel(a, b)
    expected = a + 1
    expected[:2] = -1
    expected[5:] = -1
    torch.testing.assert_close(b, expected, rtol=0, atol=0)


def test_vectorized_nonzero_min_mixed_cast_cpu():
    _check_mixed_cast_nonzero_min("c", "cpu")


@tilelang.testing.requires_cuda
def test_vectorized_nonzero_min_mixed_cast_cuda():
    _check_mixed_cast_nonzero_min("cuda", "cuda")


if __name__ == "__main__":
    tilelang.testing.main()
