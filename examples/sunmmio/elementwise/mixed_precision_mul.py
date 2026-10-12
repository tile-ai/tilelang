"""Mixed BF16/FP32 T.Tiles example with exact sunsim verification.

The execution tile is inferred from the layouts of all three shared buffers.
Each input and the output can independently use BF16 or FP32.
"""

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T


@tilelang.jit(target="sunmmio", execution_backend="sunmmio_sunsim")
def mixed_precision_mul(M=256, N=512, block_M=64, block_N=128, a_dtype="bfloat16", b_dtype="bfloat16", out_dtype="float32"):
    placement = T.placement.full_shard(0, 1)

    @T.prim_func
    def mixed_mul(
        A: T.MeshTensor((M, N), placement, a_dtype),
        B: T.MeshTensor((M, N), placement, b_dtype),
        C: T.MeshTensor((M, N), placement, out_dtype),
    ):
        with T.Kernel():
            local_m, local_n = A.local_shape
            A_shared = T.alloc_shared((block_M, block_N), a_dtype)
            B_shared = T.alloc_shared((block_M, block_N), b_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            for bm in T.serial(T.ceildiv(local_m, block_M)):
                for bn in T.serial(T.ceildiv(local_n, block_N)):
                    T.copy(A[bm * block_M, bn * block_N], A_shared)
                    T.copy(B[bm * block_M, bn * block_N], B_shared)
                    for i, j in T.Tiles(A_shared):
                        C_shared[i, j] = A_shared[i, j] * B_shared[i, j]
                    T.copy(C_shared, C[bm * block_M, bn * block_N])

    return mixed_mul


def main(M=256, N=512, block_M=64, block_N=128, working_dir=None, a_dtype="bfloat16", b_dtype="bfloat16", out_dtype="float32"):
    import ml_dtypes
    import numpy as np
    import sunsim

    dtypes = {"bfloat16": ml_dtypes.bfloat16, "float32": np.float32}
    rng = np.random.default_rng(20260922)
    a = rng.standard_normal((M, N)).astype(dtypes[a_dtype])
    b = rng.standard_normal((M, N)).astype(dtypes[b_dtype])
    placement = [sunsim.S(0), sunsim.S(1)]
    layout = sunsim.Layout.zz(block_dims=(0, 1))
    c = sunsim.Output((M, N), dtypes[out_dtype], placement=placement, layout=layout)
    kernel = mixed_precision_mul(M, N, block_M, block_N, a_dtype, b_dtype, out_dtype)
    result = kernel(
        sunsim.Input(a, placement=placement, layout=layout),
        sunsim.Input(b, placement=placement, layout=layout),
        c,
        working_dir=working_dir,
        timeout=240.0,
    )
    # Mixed operands promote to FP32; BF16 * BF16 rounds before the output cast.
    product = a.astype(np.float32) * b.astype(np.float32)
    if a_dtype == b_dtype == "bfloat16":
        product = product.astype(ml_dtypes.bfloat16)
    np.testing.assert_array_equal(c.data, product.astype(dtypes[out_dtype]))
    assert result.exit_code == 0
    print(f"PASS: {a_dtype} * {b_dtype} -> {out_dtype}, shape=({M}, {N})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=256)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--working-dir", type=Path)
    parser.add_argument("--a-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--b-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--out-dtype", choices=("bfloat16", "float32"), default="float32")
    args = parser.parse_args()
    main(args.m, args.n, args.block_m, args.block_n, args.working_dir, args.a_dtype, args.b_dtype, args.out_dtype)
