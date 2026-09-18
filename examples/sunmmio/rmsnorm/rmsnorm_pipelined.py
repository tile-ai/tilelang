"""RMSNorm on Sunmmio with fully explicit scope, layout and pipelining.

This is the "transparent" counterpart to ``rmsnorm.py``: instead of letting the
compiler infer SRAM scopes/layouts and rely on ``T.Pipelined`` for the software
pipeline, every buffer states its scope and layout, and the N loops are
hand-pipelined with explicit double buffering. The algorithm is identical to
``rmsnorm.py``.

Double buffering (manual): the X tiles live in a 2-stage RSRAM buffer
``X_db[2]``. Each iteration prefetches tile ``by+1`` into one stage (an async
DMA) while the tile/vector unit consumes tile ``by`` in the other stage, so the
load of the next tile overlaps the compute on the current one. The scale-write
loop additionally double-buffers the output ``Y_db[2]`` so an in-flight store of
tile ``by`` is not clobbered by the scale of tile ``by+1``. The actual
hardware-unit synchronization is inserted later by ``InjectSunmmioSync`` (after
this lower-and-legalize stage); here we only express the staged structure.
"""

import argparse
from typing import Callable

import tilelang
import tilelang.language as T
from tilelang import tvm as tvm
from tilelang.engine.phase import LowerAndLegalize
from tilelang.utils.target import determine_target
from tilelang.layout import make_zz_layout, make_aligned_row_major


def rmsnorm_kernel(M, N, block_M, block_N, dtype: T.dtype = T.bfloat16, eps: float = 1e-12) -> "Callable":
    zz_layout = make_zz_layout((M, N))
    placement = T.placement.full_shard(0, 1)

    accum_dtype = T.float32

    @T.prim_func
    def main(
        X: T.MeshTensor((M, N), placement, dtype, layout=zz_layout),
        Y: T.MeshTensor((M, N), placement, dtype, layout=zz_layout),
    ):
        with T.Kernel() as (_cid):
            sharded_M, sharded_N = X.local_shape

            # --- Explicit scope: an elementwise + reduction kernel with no
            # tensor-core GEMM, so every on-chip buffer lives in RSRAM (the
            # tile/vector unit's working memory). X_db/Y_db carry a leading
            # 2-stage axis for double buffering. ---
            X_db = T.alloc_shared((2, block_M, block_N), dtype, scope="shared.rsram")
            Y_db = T.alloc_shared((2, block_M, block_N), dtype, scope="shared.rsram")
            x_sq = T.alloc_shared((block_M, block_N), accum_dtype, scope="shared.rsram")
            tile_sumsq = T.alloc_shared((block_M,), accum_dtype, scope="shared.rsram")
            local_sumsq = T.alloc_shared((block_M,), accum_dtype, scope="shared.rsram")
            sumsq_dist = T.alloc_shared((T.ncols(), block_M), accum_dtype, scope="shared.rsram")
            total_sumsq = T.alloc_shared((block_M,), accum_dtype, scope="shared.rsram")
            inv_rms = T.alloc_shared((block_M,), accum_dtype, scope="shared.rsram")

            # --- Explicit layout, chosen to follow the data flow:
            #   * X_db/Y_db/x_sq carry [block_M, block_N] tiles that cross the DMA
            #     boundary with the ZZ-laid-out global X/Y (and ZZ matches the
            #     tile unit's 32x32 blocks), so they are ZZ on the trailing two
            #     dims -> a plain DMA with no layout transform.
            #   * The 1D reduction vectors (and the gather buffer sumsq_dist) are
            #     not blocked, so aligned row-major.
            # These match what inference would pick; stating them just makes the
            # file self-documenting. ---
            T.annotate_layout(
                {
                    X_db: make_zz_layout(X_db),
                    Y_db: make_zz_layout(Y_db),
                    x_sq: make_zz_layout(x_sq),
                    tile_sumsq: make_aligned_row_major(tile_sumsq, accum_dtype),
                    local_sumsq: make_aligned_row_major(local_sumsq, accum_dtype),
                    sumsq_dist: make_aligned_row_major(sumsq_dist, accum_dtype),
                    total_sumsq: make_aligned_row_major(total_sumsq, accum_dtype),
                    inv_rms: make_aligned_row_major(inv_rms, accum_dtype),
                }
            )

            nt = T.ceildiv(sharded_N, block_N)

            for bx in T.serial(T.ceildiv(sharded_M, block_M)):
                T.fill(local_sumsq, 0)

                # --- Pass 1: sum of squares, X double-buffered. ---
                # Prologue: load the first tile into stage 0.
                T.copy(X[bx * block_M : (bx + 1) * block_M, 0:block_N], X_db[0, :, :])
                for by in T.serial(nt):
                    # Prefetch the next tile into the other stage (async DMA),
                    # overlapping the square+reduce on the current stage.
                    if by + 1 < nt:
                        T.copy(X[bx * block_M : (bx + 1) * block_M, (by + 1) * block_N : (by + 2) * block_N], X_db[(by + 1) % 2, :, :])
                    for i, j in T.Tiles([block_M, block_N]):
                        x_sq[i, j] = X_db[by % 2, i, j].astype(accum_dtype) * X_db[by % 2, i, j].astype(accum_dtype)
                    T.reduce_sum(x_sq, tile_sumsq, dim=-1, clear=True)
                    for i in T.Tiles([block_M]):
                        local_sumsq[i] = local_sumsq[i] + tile_sumsq[i]

                # Combine partial sums across the row (the N-sharding axis).
                T.comm.all_gather(local_sumsq, sumsq_dist, direction="h")
                T.reduce_sum(sumsq_dist, total_sumsq, dim=0, clear=True)

                for i in T.Tiles([block_M]):
                    inv_rms[i] = T.rsqrt(total_sumsq[i] / N + eps)

                # --- Pass 2: scale and write, X and Y double-buffered. ---
                T.copy(X[bx * block_M : (bx + 1) * block_M, 0:block_N], X_db[0, :, :])
                for by in T.serial(nt):
                    if by + 1 < nt:
                        T.copy(X[bx * block_M : (bx + 1) * block_M, (by + 1) * block_N : (by + 2) * block_N], X_db[(by + 1) % 2, :, :])
                    for i, j in T.Tiles([block_M, block_N]):
                        Y_db[by % 2, i, j] = X_db[by % 2, i, j] * inv_rms[i]
                    # Store the current output stage (async DMA); the next
                    # iteration scales into the other stage, so this store is
                    # free to drain in the background.
                    T.copy(Y_db[by % 2, :, :], Y[bx * block_M : (bx + 1) * block_M, by * block_N : (by + 1) * block_N])

    return main


def main(M, N) -> None:
    target = determine_target("Sunmmio", return_object=True)

    pass_configs = {tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True}
    with tvm.target.Target(target), tvm.transform.PassContext(config=pass_configs):
        kernel = rmsnorm_kernel(M, N, block_M=128, block_N=128, dtype=T.bfloat16)
        mod = LowerAndLegalize(tvm.IRModule({"main": kernel}), target)
        print(mod)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=1024)
    args, _ = parser.parse_known_args()
    main(args.m, args.n)
