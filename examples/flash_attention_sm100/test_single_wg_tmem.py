"""Sanity test: single-WG TMEM round-trip (gemm anchor + chunked
write + chunked read). If this fails too, the layout issue is
unrelated to cross-WG. If it works, cross-WG is the specific bug.
"""
import argparse
import tilelang
import tilelang.language as T


PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    "tl.disable_thread_storage_sync": True,
}


@tilelang.jit(out_idx=[0], pass_configs=PASS_CFG, target={"kind": "cuda", "arch": "sm_100"})
def single_wg_tmem(M: int = 64, N: int = 128, K: int = 64):
    accum_dtype = T.float32
    dtype = T.bfloat16
    PATTERN = 7.0
    CHUNK = 32

    @T.prim_func
    def main(Output: T.Tensor([M, N], accum_dtype)):
        with T.Kernel(1, 1, 1, threads=128) as (bx, by, bz):
            A_smem = T.alloc_shared([M, K], dtype)
            B_smem = T.alloc_shared([N, K], dtype)
            X_tmem = T.alloc_tmem([M, N], accum_dtype)
            O_chunk = T.alloc_fragment([M, CHUNK], accum_dtype)
            O_full = T.alloc_fragment([M, N], accum_dtype)
            mbar_mma = T.alloc_barrier(1)

            T.fill(A_smem, T.Cast(dtype, 0))
            T.fill(B_smem, T.Cast(dtype, 0))
            T.tcgen05_gemm(
                A_smem, B_smem, X_tmem,
                transpose_B=True, mbar=mbar_mma, clear_accum=True,
            )
            T.mbarrier_wait_parity(mbar_mma, 0)

            # Write PATTERN
            T.fill(O_chunk, PATTERN)
            T.copy(O_chunk, X_tmem[:, 0:32])
            T.copy(O_chunk, X_tmem[:, 32:64])
            T.copy(O_chunk, X_tmem[:, 64:96])
            T.copy(O_chunk, X_tmem[:, 96:128])

            # Read back
            T.copy(X_tmem[:, 0:32], O_chunk)
            for i, j in T.Parallel(M, CHUNK):
                O_full[i, j] = O_chunk[i, j]
            T.copy(X_tmem[:, 32:64], O_chunk)
            for i, j in T.Parallel(M, CHUNK):
                O_full[i, 32 + j] = O_chunk[i, j]
            T.copy(X_tmem[:, 64:96], O_chunk)
            for i, j in T.Parallel(M, CHUNK):
                O_full[i, 64 + j] = O_chunk[i, j]
            T.copy(X_tmem[:, 96:128], O_chunk)
            for i, j in T.Parallel(M, CHUNK):
                O_full[i, 96 + j] = O_chunk[i, j]
            T.copy(O_full, Output)

    return main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=64)
    args = parser.parse_args()
    out = single_wg_tmem(M=args.M)().cpu()
    n_total = out.numel()
    n_pattern = (out == 7.0).sum().item()
    print(f"shape={tuple(out.shape)}  pattern_hits={n_pattern}/{n_total}")
    print(f"  sample [0,:8]: {out[0,:8].tolist()}")
    if n_pattern == n_total:
        print("PASS: single-WG TMEM round-trip works.")
    else:
        print("FAIL: even single-WG breaks — layout issue unrelated to cross-WG.")


if __name__ == "__main__":
    main()
