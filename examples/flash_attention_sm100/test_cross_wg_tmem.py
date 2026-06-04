"""Minimal repro for cross-WG TMEM handoff bug.

Mirrors the FA structure: one shared O_chunk fragment used by both WGs
(layout inferred from union of uses), one TMEM allocation anchored by
a tcgen05_gemm in WG0.

  WG0 (tid 0-127):   tcgen05_gemm(zeros, zeros) → X_tmem = 0 (anchor).
                     Then chunked overwrite of X_tmem with PATTERN via
                     tcgen05_st on the shared O_chunk fragment.
                     arrive mbar_handoff.
  WG1 (tid 128-255): wait mbar_handoff. Chunked read X_tmem via
                     tcgen05_ld on the shared O_chunk fragment, dump
                     to GMEM.

Expected (if cross-WG TMEM works): GMEM == PATTERN (7.0) everywhere.
Observed in FA: GMEM stays at 0 (MMA output untouched).
"""
import argparse

import torch
import tilelang
import tilelang.language as T


PASS_CFG = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    "tl.disable_thread_storage_sync": True,
}


@tilelang.jit(out_idx=[0], pass_configs=PASS_CFG, target={"kind": "cuda", "arch": "sm_100"})
def cross_wg_tmem_kernel(M: int = 64, N: int = 128, K: int = 64):
    accum_dtype = T.float32
    dtype = T.bfloat16
    PATTERN = 7.0
    CHUNK = 32

    @T.prim_func
    def main(Output: T.Tensor([M, N], accum_dtype)):
        with T.Kernel(1, 1, 1, threads=256) as (bx, by, bz):
            # P_tmem allocated to use as the A operand for TS gemm
            # (matching FA's PV: P from TMEM, V from SMEM, O to TMEM).
            P_tmem_anchor = T.alloc_tmem([M, K], dtype)
            B_smem = T.alloc_shared([N, K], dtype)
            X_tmem = T.alloc_tmem([M, N], accum_dtype)
            # ONE shared fragment used by both WGs (matches FA's O_chunk).
            O_chunk = T.alloc_fragment([M, CHUNK], accum_dtype)
            O_full = T.alloc_fragment([M, N], accum_dtype)

            mbar_mma = T.alloc_barrier(1)
            mbar_handoff = T.alloc_barrier(128)
            tid = T.get_thread_binding()

            # WG0: anchor TMEM via TS MMA (P_tmem @ B_smem = X_tmem),
            # then overwrite chunks with PATTERN.
            if tid < 128:
                T.fill(B_smem, T.Cast(dtype, 0))
                T.tcgen05_gemm(
                    P_tmem_anchor, B_smem, X_tmem,
                    transpose_B=True, mbar=mbar_mma, clear_accum=True,
                )
                T.mbarrier_wait_parity(mbar_mma, 0)

                T.fill(O_chunk, PATTERN)
                T.copy(O_chunk, X_tmem[:, 0:32])
                T.copy(O_chunk, X_tmem[:, 32:64])
                T.copy(O_chunk, X_tmem[:, 64:96])
                T.copy(O_chunk, X_tmem[:, 96:128])
                T.mbarrier_arrive(mbar_handoff)

            # WG1: wait, read chunks of X_tmem into O_chunk, accumulate
            # into O_full, dump to GMEM.
            elif tid < 256:
                T.mbarrier_wait_parity(mbar_handoff, 0)
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
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--K", type=int, default=64)
    args = parser.parse_args()

    kernel = cross_wg_tmem_kernel(M=args.M, N=args.N, K=args.K)
    out = kernel().cpu()

    n_total = out.numel()
    n_pattern = (out == 7.0).sum().item()
    n_zero = (out == 0.0).sum().item()
    print(f"shape={tuple(out.shape)} dtype={out.dtype}")
    print(f"  == 7.0  (PATTERN, WG0 wrote): {n_pattern}/{n_total}")
    print(f"  == 0.0  (MMA wrote zeros)   : {n_zero}/{n_total}")
    print(f"  min/max : {out.min().item():.4f} / {out.max().item():.4f}")
    print(f"  sample [0, :8]: {out[0, :8].tolist()}")
    print(f"  sample [{args.M//2}, :8]: {out[args.M//2, :8].tolist()}")
    print(f"  sample [{args.M-1}, :8]: {out[args.M-1, :8].tolist()}")

    print()
    if n_pattern == n_total:
        print("PASS: cross-WG TMEM handoff works (WG1 sees WG0's tcgen05_st writes).")
    elif n_zero == n_total:
        print("FAIL: WG1 reads MMA's zeros — WG0's chunked writes invisible cross-WG.")
        print("      This is the bug blocking the FA 6-role split.")
    else:
        print("MIXED — partial visibility, suggests layout/row mismatch.")


if __name__ == "__main__":
    main()
