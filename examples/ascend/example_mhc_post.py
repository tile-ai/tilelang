"""MHC post forward on Ascend NPU.

x = c * d + a^T @ b, tiled over the hidden dimension with a pipelined
double-buffered copy. The AutoSimtVF pass promotes the `T.Parallel` loop.
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
            x_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            b_shared = T.alloc_shared((mhc, h_blk), T.bfloat16)
            d_shared = T.alloc_shared(h_blk, T.bfloat16)

            x_local = T.alloc_fragment((mhc, h_blk), T.float32)
            b_local = T.alloc_fragment((mhc, h_blk), T.float32)
            d_local = T.alloc_fragment(h_blk, T.float32)

            a_local = T.alloc_fragment((mhc, mhc), T.float32)
            c_local = T.alloc_fragment(mhc, T.float32)
            T.copy(a[pid_n, 0, 0], a_local)
            T.copy(c[pid_n, 0], c_local)

            for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
                T.copy(b[pid_n, 0, i0_h * h_blk], b_shared)
                T.copy(d[pid_n, i0_h * h_blk], d_shared)

                T.copy(b_shared, b_local)
                T.copy(d_shared, d_local)
                for i_mhco, i1_h in T.Parallel(mhc, h_blk):
                    x_local[i_mhco, i1_h] = c_local[i_mhco] * d_local[i1_h]
                    for i_mhci in T.serial(mhc):
                        x_local[i_mhco, i1_h] += a_local[i_mhci, i_mhco] * b_local[i_mhci, i1_h]
                T.copy(x_local, x_shared)

                T.copy(x_shared, x[pid_n, 0, i0_h * h_blk])

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
