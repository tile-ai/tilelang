"""Regression tests for MTE copy with small / strided buffers.

These reproduce bugs found in mHC (multi-Head Capsule) kernels where:
1. Sinkhorn: GM→UB copy of (1,4,4) float32 has row_bytes=16 < 32B burst minimum.
2. Expand:   UB→GM copy into a 3D output with non-trivial stride between slices.

Both should pass once the copy lowering correctly coalesces contiguous dimensions
and handles sub-burst row sizes.
"""

import pytest
import torch
import tilelang
from tilelang.ascend import language as T


# ─── Sinkhorn-style: small matrix copy + reduce ─────────────────────────────


@tilelang.jit
def _sinkhorn_fwd_kernel(hidden_size: int, token_block_size: int, repeat: int, eps: float):
    """Minimal sinkhorn forward that exercises GM→UB→GM copy of (N, H, H) float32.

    Uses the Ascend pattern: GM→UB (shared) → SimtVF (fragment) → UB → GM.
    """
    num_tokens = T.dynamic("num_tokens")

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
        out: T.Tensor[(num_tokens, hidden_size, hidden_size), T.float32],
    ) -> None:
        with T.Kernel(T.ceildiv(num_tokens, token_block_size)) as pid:
            ub = T.alloc_shared((token_block_size, hidden_size, hidden_size), T.float32)
            T.copy(x[pid * token_block_size, 0, 0], ub)

            with T.SimtVF(threads=256):
                frag = T.alloc_fragment((token_block_size, hidden_size, hidden_size), T.float32)
                row_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)
                col_sum = T.alloc_fragment((token_block_size, hidden_size), T.float32)

                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    frag[i, j, k] = ub[i, j, k]

                # softmax(-1) + eps
                row_max = T.alloc_fragment((token_block_size, hidden_size), T.float32)
                T.reduce_max(frag, row_max, dim=2)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    frag[i, j, k] = T.exp(frag[i, j, k] - row_max[i, j])
                T.reduce_sum(frag, row_sum, dim=2)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    frag[i, j, k] = frag[i, j, k] / row_sum[i, j] + eps

                # comb / (comb.sum(-2) + eps)
                T.reduce_sum(frag, col_sum, dim=1)
                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    frag[i, j, k] = frag[i, j, k] / (col_sum[i, k] + eps)

                for _ in T.serial(repeat - 1):
                    T.reduce_sum(frag, row_sum, dim=2)
                    for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                        frag[i, j, k] = frag[i, j, k] / (row_sum[i, j] + eps)

                    T.reduce_sum(frag, col_sum, dim=1)
                    for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                        frag[i, j, k] = frag[i, j, k] / (col_sum[i, k] + eps)

                for i, j, k in T.Parallel(token_block_size, hidden_size, hidden_size):
                    ub[i, j, k] = frag[i, j, k]

            T.copy(ub, out[pid * token_block_size, 0, 0])

    return kernel


def _sinkhorn_ref(x: torch.Tensor, repeat: int, eps: float) -> torch.Tensor:
    """Pure PyTorch reference for sinkhorn normalization."""
    comb = x.clone()
    comb = comb.softmax(-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return comb


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("hidden_size", [4])
def test_mhc_sinkhorn_copy(target, num_tokens, hidden_size):
    """Test GM→UB→GM copy with (N, 4, 4) float32 — row_bytes=16 < 32."""
    repeat = 3
    eps = 1e-6
    token_block_size = 1

    prim = _sinkhorn_fwd_kernel.get_tir(hidden_size, token_block_size, repeat, eps)
    kernel = tilelang.compile(prim, target=target)

    device = torch.device("npu")
    x = torch.randn(num_tokens, hidden_size, hidden_size, dtype=torch.float32, device=device)
    out = torch.empty_like(x)
    kernel(x, out)
    torch.npu.synchronize()

    expected = _sinkhorn_ref(x, repeat, eps)
    torch.testing.assert_close(out, expected, atol=1e-4, rtol=1e-4)


# ─── Expand-style: strided UB→GM copy ───────────────────────────────────────


@tilelang.jit
def _expand_fwd_kernel(hidden: int, mhc_mult: int):
    """Expand x[n,h] → o[n,mhc,h] by broadcasting along mhc dimension.

    Exercises UB→GM copy where dst buffer has stride between slices
    (output is [n, mhc, h], copying a [1, h] UB slice into each mhc slot).
    """
    num_tokens = T.dynamic("num_tokens")
    h = hidden
    mhc = mhc_mult

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, h), T.bfloat16],
        o: T.Tensor[(num_tokens, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(num_tokens) as pid:
            x_ub = T.alloc_shared((h,), T.bfloat16)
            T.copy(x[pid, 0], x_ub)
            for m in T.serial(mhc):
                T.copy(x_ub, o[pid, m, 0])

    return kernel


def _expand_ref(x: torch.Tensor, mhc: int) -> torch.Tensor:
    """Pure PyTorch reference: expand x[n,h] → o[n,mhc,h]."""
    return x.unsqueeze(1).expand(-1, mhc, -1).contiguous()


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
@pytest.mark.parametrize("num_tokens", [1, 8])
@pytest.mark.parametrize("hidden", [128, 256])
@pytest.mark.parametrize("mhc_mult", [2, 4])
def test_mhc_expand_copy(target, num_tokens, hidden, mhc_mult):
    """Test UB→GM copy into strided 3D output buffer."""
    prim = _expand_fwd_kernel.get_tir(hidden, mhc_mult)
    kernel = tilelang.compile(prim, target=target)

    device = torch.device("npu")
    x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device=device)
    out = torch.empty(num_tokens, mhc_mult, hidden, dtype=torch.bfloat16, device=device)
    kernel(x, out)
    torch.npu.synchronize()

    expected = _expand_ref(x, mhc_mult)
    torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    print("=== Sinkhorn copy tests ===")
    for n in [1, 4]:
        test_mhc_sinkhorn_copy("ascend", n, 4)
        print(f"  PASS: num_tokens={n}, hidden_size=4")

    print("=== Expand copy tests ===")
    for n in [1, 8]:
        for h in [128, 256]:
            for mhc in [2, 4]:
                test_mhc_expand_copy("ascend", n, h, mhc)
                print(f"  PASS: num_tokens={n}, hidden={h}, mhc={mhc}")
