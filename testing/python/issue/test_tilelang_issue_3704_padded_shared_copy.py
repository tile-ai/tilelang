"""Regression test for issue #3704.

A `T.copy` from a non-16-aligned global region (K=100) into a padded shared
buffer (BK=112) used to be classified as a TMA bulk copy and then hard-fail in
`Copy::LowerBulk` with `Shared tile and global tile shape mismatch`.  The
selection now rejects incompatible extents so the copy lowers through the
element-wise/`cp.async` path; only lowering is exercised here.
"""

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

BT, BS, BV = 128, 64, 64
K = 100
BK = (K + 15) // 16 * 16


def _kernel():
    @T.prim_func
    def main(
        q: T.Tensor((1, 256, 4, K), "float32"),
        k: T.Tensor((1, 256, 4, K), "float32"),
        v: T.Tensor((1, 256, 4, BV), "float32"),
        o: T.Tensor((1, 256, 4, BV), "float32"),
    ):
        with T.Kernel(4, T.ceildiv(256, BT), 1, threads=128) as (i_h, i_t, i_b):
            s_q = T.alloc_shared((BT, BK), "float32")
            s_k = T.alloc_shared((BS, BK), "float32")
            s_v = T.alloc_shared((BS, BV), "float32")
            b_s = T.alloc_fragment((BT, BS), "float32")
            b_p = T.alloc_shared((BT, BS), "float32")
            b_o = T.alloc_fragment((BT, BV), "float32")
            T.copy(q[i_b, i_t * BT : i_t * BT + BT, i_h, :], s_q)
            T.fill(b_o, 0)
            for k_idx in T.Pipelined(0, T.ceildiv((i_t + 1) * BT, BS), num_stages=1):
                k_s = k_idx * BS
                T.copy(k[i_b, k_s : k_s + BS, i_h, :], s_k)
                T.copy(v[i_b, k_s : k_s + BS, i_h, :], s_v)
                T.clear(b_s)
                T.gemm(s_q, s_k, b_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(BT, BS):
                    b_s[i, j] = T.if_then_else((k_s + j <= i_t * BT + i) & (k_s + j < 256), T.exp2(b_s[i, j] * 0.18 - 3.0), 0.0)
                T.copy(b_s, b_p)
                T.gemm(b_p, s_v, b_o, policy=T.GemmWarpPolicy.FullRow)
            for i, d in T.Parallel(BT, BV):
                o[i_b, i_t * BT + i, i_h, d] = b_o[i, d] / BS

    return main


@tilelang.testing.requires_cuda
def test_padded_shared_copy_lowers():
    cfg = {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    }
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    with target, tvm.transform.PassContext(config=cfg):
        tilelang.lower(_kernel(), target=target, enable_device_compile=False)


if __name__ == "__main__":
    tilelang.testing.main()
