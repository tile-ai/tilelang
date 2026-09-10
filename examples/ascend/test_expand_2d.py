"""Test for UB→GM copy with extent=1 outer dims (mHC expand with 2D UB buffer).

Exercises the case where dst range has extent=1 modes with non-trivial stride,
which requires Coalesce to filter out size-1 modes.
"""

import pytest
import torch
import tilelang
from tilelang.ascend import language as T


@tilelang.jit
def _expand_2d_kernel(hidden: int, mhc: int, h_blk: int, num_sms: int = 32):
    """Expand x[n,h] → o[n,mhc,h] with tiled hidden dim.

    UB buffer is (token_block_size, h_blk) to exercise 2D UB→GM copy
    where dst has extent=1 outer dims with large strides.
    """
    num_tokens = T.dynamic("num_tokens")
    h = hidden
    n_h_tiles = h // h_blk

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, h), T.bfloat16],
        o: T.Tensor[(num_tokens, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(num_sms) as bx:
            token_block_size = 1
            x_ub = T.alloc_shared((token_block_size, h_blk), T.bfloat16)

            total_tiles = num_tokens * n_h_tiles
            for tile_idx in T.Persistent([total_tiles], num_sms, bx):
                pid_tok = tile_idx // n_h_tiles
                pid_h = tile_idx % n_h_tiles
                T.copy(x[pid_tok * token_block_size, pid_h * h_blk], x_ub)
                for mi in T.serial(mhc):
                    T.copy(
                        x_ub,
                        o[pid_tok * token_block_size, mi, pid_h * h_blk],
                    )

    return kernel


@pytest.mark.parametrize("target", ["ascend"])
def test_expand_2d(target):
    device = torch.device("npu")
    hidden, mhc, h_blk = 4096, 4, 128
    num_tokens = 8
    prim = _expand_2d_kernel.get_tir(hidden, mhc, h_blk)
    kernel = tilelang.compile(prim, target=target)
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    out = torch.empty(num_tokens, mhc, hidden, dtype=torch.bfloat16, device=device)
    kernel(x, out)
    torch.npu.synchronize()
    expected = x.unsqueeze(1).expand(-1, mhc, -1).contiguous()
    diff = (out - expected).abs().max().item()
    print(f"max_diff={diff:.2e}")
    assert diff < 1e-2, f"max_diff={diff:.2e}"


if __name__ == "__main__":
    for target in ("ascend",):
        test_expand_2d(target)
        print(f"PASS ({target})")
