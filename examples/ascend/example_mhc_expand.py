"""MHC expand forward/backward on Ascend NPU.

Broadcasts `x[n, h]` to `o[n, mhc, h]` and reduces the gradient back. Written
in plain TileLang; the AutoSimtVF pass promotes the `T.Parallel` loops.
"""

import torch
import tilelang
import tilelang.ascend.language as T


def mhc_expand_fwd(hidden, mhc_mult):
    n = T.dynamic("num_tokens")
    h = hidden
    mhc = mhc_mult
    blk_n = 32
    blk_h = 128
    num_hidden_blocks = (h + blk_h - 1) // blk_h

    @T.prim_func
    def main(
        x: T.Tensor[(n, h), T.bfloat16],
        o: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(T.ceildiv(n, blk_n) * num_hidden_blocks) as pid:
            pid_i = pid // num_hidden_blocks
            pid_j = pid % num_hidden_blocks
            if n > 0:
                xl = T.alloc_fragment((blk_n, blk_h), T.bfloat16)
                T.copy(x[pid_i * blk_n, pid_j * blk_h], xl)
                for m in T.serial(mhc):
                    for ti, tj in T.Parallel(blk_n, blk_h):
                        i = pid_i * blk_n + ti
                        j = pid_j * blk_h + tj
                        if i < n and j < h:
                            o[i, m, j] = xl[ti, tj]

    return main


def mhc_expand_bwd(hidden, mhc_mult):
    n = T.dynamic("num_tokens")
    h = hidden
    mhc = mhc_mult
    blk_n = 32
    blk_h = 128
    num_hidden_blocks = (h + blk_h - 1) // blk_h

    @T.prim_func
    def main(
        o_grad: T.Tensor[(n, mhc, h), T.bfloat16],
        x_grad: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(T.ceildiv(n, blk_n) * num_hidden_blocks) as pid:
            pid_i = pid // num_hidden_blocks
            pid_j = pid % num_hidden_blocks
            if n > 0:
                xgl = T.alloc_fragment((blk_n, blk_h), T.float32)
                T.fill(xgl, 0)
                for m in T.serial(mhc):
                    for ti, tj in T.Parallel(blk_n, blk_h):
                        i = pid_i * blk_n + ti
                        j = pid_j * blk_h + tj
                        if i < n and j < h:
                            xgl[ti, tj] += o_grad[i, m, j]
                T.copy(xgl, x_grad[pid_i * blk_n, pid_j * blk_h])

    return main


def ref_program_expand_fwd(x, mhc_mult):
    return x.unsqueeze(-2).expand(*x.shape[:-1], mhc_mult, x.shape[-1]).contiguous()


def ref_program_expand_bwd(o_grad):
    return o_grad.float().sum(dim=1).to(torch.bfloat16)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("npu")
    num_tokens, hidden, mhc_mult = 128, 1280, 4

    x = torch.randn((num_tokens, hidden), device=device, dtype=torch.bfloat16)
    kernel = tilelang.compile(mhc_expand_fwd(hidden, mhc_mult), target="ascend", out_idx=-1)
    actual = kernel(x)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, ref_program_expand_fwd(x, mhc_mult))
    print("PASS: mhc_expand_fwd")

    o_grad = torch.randn((num_tokens, mhc_mult, hidden), device=device, dtype=torch.bfloat16)
    kernel = tilelang.compile(mhc_expand_bwd(hidden, mhc_mult), target="ascend", out_idx=-1)
    x_grad = kernel(o_grad)
    torch.npu.synchronize()
    torch.testing.assert_close(x_grad, ref_program_expand_bwd(o_grad), rtol=1e-5, atol=2e-5)
    print("PASS: mhc_expand_bwd")
