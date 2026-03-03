import torch

import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.jit
def _issue1846_fill_scalar(fill, M):
    X = T.empty((M), dtype="float32")
    with T.Kernel(1, threads=M) as _:
        for i in T.Parallel(M):
            X[i] = fill
    return X


@tilelang.testing.requires_cuda
def test_issue_1846_eager_jit_call_executes():
    """Regression test for issue #1846.

    Calling an eager-style @tilelang.jit function (builder pattern with T.empty)
    should compile and execute the kernel, returning the output tensor directly.
    """

    M = 32
    fill = 1.0

    out = _issue1846_fill_scalar(fill=fill, M=M)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (M,)
    torch.testing.assert_close(out, torch.full((M,), fill, device=out.device, dtype=out.dtype))


if __name__ == "__main__":
    tilelang.testing.main()
