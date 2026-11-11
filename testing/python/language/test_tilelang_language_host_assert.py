# Copyright (c) Tile-AI Corporation. All Rights Reserved.
"""Run-time behavior of a host-side T.Assert: the error it raises and when it raises it."""
# ruff: noqa

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
def test_host_assert_raises_the_requested_error_kind():
    """The caller receives the error kind the assertion names, not a fixed one."""
    import torch

    @tilelang.jit
    def same_length():
        m = T.dynamic("m")
        n = T.dynamic("n")

        @T.prim_func
        def main(x: T.Tensor[(m,), T.float32], y: T.Tensor[(n,), T.float32]):
            with T.Assert(m == n, "x and y must have the same length", "ValueError"):
                with T.Kernel(T.ceildiv(m, 128), threads=128) as bx:
                    tx = T.get_thread_binding()
                    i = bx * 128 + tx
                    if i < m:
                        y[i] = x[i] * 2.0

        return main

    kernel = same_length()
    x = torch.arange(256, dtype=torch.float32, device="cuda")
    y = torch.empty(256, dtype=torch.float32, device="cuda")
    kernel(x, y)
    torch.testing.assert_close(y, x * 2.0)

    # ValueError is not a RuntimeError, so this cannot pass on a host that
    # raises every assertion as RuntimeError. The rejected call must not have
    # launched either: its output stays as it was.
    probe = torch.full((255,), -1.0, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="x and y must have the same length"):
        kernel(x, probe)
    torch.cuda.synchronize()
    assert bool((probe == -1.0).all())


if __name__ == "__main__":
    tilelang.testing.main()
