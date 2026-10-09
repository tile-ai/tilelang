"""MHC post forward on Ascend NPU.

x = c * d + a^T @ b, tiled over the hidden dimension with a pipelined
double-buffered copy. The AutoSimtVF pass promotes the `T.Parallel` loop.

CUDA reference: TileKernels ``tile_kernels/mhc/post_cuda.py``.
"""

import math

import torch
import tilelang
import tilelang.ascend.language as T


def mhc_post_fwd(mhc, hidden, h_blk=1024):
    n = T.dynamic("num_tokens")
    h = hidden
    h_blk = math.gcd(hidden, h_blk)

    @T.prim_func
    def main(
        a: T.Tensor[(n, mhc, mhc), T.float32],
        b: T.Tensor[(n, mhc, h), T.bfloat16],
        c: T.Tensor[(n, mhc), T.float32],
        d: T.Tensor[(n, h), T.bfloat16],
        x: T.Tensor[(n, mhc, h), T.bfloat16],
    ) -> None:
        with T.Kernel(n) as pid_n:
            x_frag = T.alloc_fragment((mhc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)

            a_frag = T.alloc_fragment((mhc, mhc), T.float32)
            c_frag = T.alloc_fragment(mhc, T.float32)
            T.copy(a[pid_n, 0, 0], a_frag)
            T.copy(c[pid_n, 0], c_frag)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared)
                T.copy(d[pid_n, i0_h * h_blk], d_shared)

                # AutoSimtVF cannot mutate a scalar allocated outside its VF.
                # Index the per-element CUDA accumulator as a fragment instead.
                x_local = T.alloc_fragment((mhc, h_blk), T.float32)
                for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                    x_local[i_mhco, i1_h] = c_frag[i_mhco] * d_shared[i1_h]
                    for i_mhci in T.unroll(mhc):
                        x_local[i_mhco, i1_h] += a_frag[i_mhci, i_mhco] * b_shared[i_mhci, i1_h]
                    x_frag[i_mhco, i1_h] = x_local[i_mhco, i1_h]

                T.copy(x_frag, x[pid_n, 0, i0_h * h_blk])

    return main


def ref_program(a, b, c, d):
    return (c.unsqueeze(-1) * d.float().unsqueeze(1) + torch.bmm(a.transpose(1, 2), b.float())).bfloat16()


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("npu")
    num_tokens, mhc, hidden = 8, 4, 1280

    a = torch.randn((num_tokens, mhc, mhc), device=device, dtype=torch.float32)
    b = torch.randn((num_tokens, mhc, hidden), device=device, dtype=torch.bfloat16)
    c = torch.randn((num_tokens, mhc), device=device, dtype=torch.float32)
    d = torch.randn((num_tokens, hidden), device=device, dtype=torch.bfloat16)

    kernel = tilelang.compile(mhc_post_fwd(mhc, hidden), target="ascend", out_idx=-1)

    actual = kernel(a, b, c, d)
    torch.npu.synchronize()
    torch.testing.assert_close(actual, ref_program(a, b, c, d), rtol=1e-2, atol=1e-2)
    print("PASS: mhc_post_fwd")
