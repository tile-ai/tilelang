"""MHC pre-apply-mix forward on Ascend NPU.

o = sum_mhc(mix * x), with one CTA per token and hidden tile. Written
in plain TileLang; the AutoSimtVF pass promotes the `T.Parallel` loop.

CUDA reference: TileKernels ``tile_kernels/mhc/pre_apply_mix_cuda.py``.
"""

import math

import torch
import tilelang
import tilelang.ascend.language as T


def mhc_pre_apply_mix_fwd(mhc_mult, hidden, h_blk=1024):
    n = T.dynamic("n")
    h = hidden
    mhc = mhc_mult
    h_blk = math.gcd(h_blk, hidden)
    h_ctas = h // h_blk

    @T.prim_func
    def main(
        x: T.Tensor[(n, mhc, h), T.bfloat16],
        mix: T.Tensor[(n, mhc), T.float32],
        o: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n * h_ctas) as pid:
            pid_n = pid // h_ctas
            pid_h = pid % h_ctas
            mixl = T.alloc_fragment(mhc, T.float32)
            T.copy(mix[pid_n, 0], mixl)

            xs = T.alloc_shared((mhc, h_blk), T.bfloat16)
            xl = T.alloc_fragment((mhc, h_blk), T.float32)
            T.copy(x[pid_n, 0, pid_h * h_blk], xs)
            T.copy(xs, xl)

            os = T.alloc_shared(h_blk, T.bfloat16)
            ol = T.alloc_fragment(h_blk, T.float32)
            T.clear(ol)

            for i_mhc in T.serial(mhc):
                for i1_h in T.Parallel(h_blk):
                    ol[i1_h] += mixl[i_mhc] * xl[i_mhc, i1_h]

            T.copy(ol, os)
            T.copy(os, o[pid_n, pid_h * h_blk])

    return main


def mhc_pre_apply_mix_fwd_stage2(mhc_mult, hidden, h_blk=1024, threads=128):
    n = T.dynamic("n")
    h = hidden
    mhc = mhc_mult
    h_blk = math.gcd(h_blk, hidden)
    h_ctas = h // h_blk

    @T.prim_func
    def main(
        x: T.Tensor[(n, mhc, h), T.bfloat16],
        mix: T.Tensor[(n, mhc), T.float32],
        o: T.Tensor[(n, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n * h_ctas) as pid:
            pid_n = pid // h_ctas
            pid_h = pid % h_ctas
            xs = T.alloc_shared((mhc, h_blk), T.bfloat16)
            os = T.alloc_shared(h_blk, T.bfloat16)
            T.copy(x[pid_n, 0, pid_h * h_blk], xs)

            with T.SimtVF(threads=threads):
                mixl = T.alloc_fragment(mhc, T.float32)
                T.copy(mix[pid_n, 0], mixl)
                xl = T.alloc_fragment((mhc, h_blk), T.float32)
                T.copy(xs, xl)
                ol = T.alloc_fragment(h_blk, T.float32)
                T.clear(ol)
                for i_mhc in T.serial(mhc):
                    for i1_h in T.Parallel(h_blk):
                        ol[i1_h] += mixl[i_mhc] * xl[i_mhc, i1_h]
                T.copy(ol, os)
            T.copy(os, o[pid_n, pid_h * h_blk])

    return main


def ref_program(x, mix):
    return (mix.unsqueeze(-1).float() * x.float()).sum(dim=1).to(torch.bfloat16)


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("npu")
    num_tokens, mhc_mult, hidden = 128, 4, 1280

    x = torch.randn((num_tokens, mhc_mult, hidden), device=device, dtype=torch.bfloat16)
    mix = torch.randn((num_tokens, mhc_mult), device=device, dtype=torch.float32)

    kernel = tilelang.compile(
        mhc_pre_apply_mix_fwd(mhc_mult, hidden),
        target="ascend",
        out_idx=-1,
    )
    actual = kernel(x, mix)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, ref_program(x, mix), rtol=1e-2, atol=1e-2)
    print("PASS: mhc_pre_apply_mix_fwd")
