"""Correctness coverage for UB fills outside a vector function.

A fill that is not enclosed by a VF block is element-wise scalar work: it runs
on the scalar core, and the vector constructors a vectorized broadcast would
need are simt-only and rejected by bisheng there.
"""

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
import torch


@tilelang.jit(target="ascend", out_idx=[0])
def _constant_fill_kernel(rows: int, cols: int, value: float):
    @T.prim_func
    def _constant_fill(out: T.Tensor((rows, cols), "float32")):
        with T.Kernel(1):
            buf = T.alloc_shared((rows, cols), "float32")
            T.fill(buf, T.float32(value))
            T.copy(buf, out)

    return _constant_fill


@tilelang.jit(target="ascend", out_idx=[1])
def _scalar_fill_kernel(iters: int, tile: int):
    @T.prim_func
    def _scalar_fill(
        x: T.Tensor((iters,), "float32"),
        out: T.Tensor((iters, tile), "float32"),
    ):
        with T.Kernel(1):
            buf = T.alloc_shared((tile,), "float32")
            for i in T.serial(iters):
                T.fill(buf, x[i])
                T.copy(buf, out[i, :])

    return _scalar_fill


def test_constant_fill_out_of_vf():
    rows, cols, value = 2, 32, -1.0
    out = _constant_fill_kernel(rows, cols, value)()
    torch.npu.synchronize()

    expected = torch.full((rows, cols), value, dtype=torch.float32)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_scalar_fill_out_of_vf():
    iters, tile = 3, 64
    device = torch.device("npu")
    x = torch.arange(iters, dtype=torch.float32, device=device)
    out = _scalar_fill_kernel(iters, tile)(x)
    torch.npu.synchronize()

    expected = x.repeat_interleave(tile).reshape(iters, tile)
    torch.testing.assert_close(out.cpu(), expected.cpu(), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
