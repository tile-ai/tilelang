"""Example: UB merge with Cube+Vector on Ascend NPU.

Demonstrates the Ascend MergeUBAllocations pass reusing two T.alloc_shared
staging buffers (buf_a, buf_b) whose lifetimes are serialized across the
Cube (AIC) producer and the Vector (AIV) consumer via cross-core flags.

Cross-core flag protocol (per AIV sid=0/1; flag base offset is sid*16):
  - AIV->AIC flag 4+sid*16: "AIV ready for next dual_copy" (initial; re-set after each copy out)
  - AIC->AIV flag 6+sid*16: "buf_a is ready in UB"
  - AIC->AIV flag 8+sid*16: "buf_b is ready in UB"

The buf_a-consumed handshake (AIV signals flag 4 after copying buf_a to GM)
gates the AIC's dual_copy to buf_b, so buf_a's lifetime ends before buf_b's
lifetime begins. This lets the UB merge pass place buf_a and buf_b at the same
UB offset (verify by inspecting the generated `.asc`).

dual_copy splits the L0C tile across the two AIVs along the M dimension:
each AIV sees its half in its own UB segment as (TILE_M // 2, TILE_N).
"""

import argparse

import tilelang
import tilelang.language as T
import torch


def gemm_ub_merge(M=256, K=256, N=128):
    TILE_M = 128
    TILE_N = 128
    TILE_K = 128
    K_TILES = K // TILE_K
    HALF_M = TILE_M // 2  # per-AIV M-half for dual_copy

    @T.prim_func
    def main(
        A: T.Buffer((M, K), "float16"),
        B: T.Buffer((N, K), "float16"),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1) as _:
            # Per-AIV halves: dual_copy M-splits the (TILE_M, TILE_N) L0C tile
            # so each AIV writes (HALF_M, TILE_N) into its own UB segment.
            buf_a = T.alloc_shared((HALF_M, TILE_N), "float32")
            buf_b = T.alloc_shared((HALF_M, TILE_N), "float32")

            a_l1 = T.alloc_l1((TILE_M, TILE_K), "float16")
            b_l1 = T.alloc_l1((TILE_N, TILE_K), "float16")
            acc = T.alloc_l0c((TILE_M, TILE_N), "float32")

            with T.Cube():
                T.ascend_set_flag("MTE1_MTE2", 0)
                T.ascend_set_flag("FIX_M", 0)

                # Tile 0: A[0:128] @ B^T -> acc -> buf_a (L0C -> UB, M-split).
                for kt in T.Serial(K_TILES):
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[0:TILE_M, kt * TILE_K : (kt + 1) * TILE_K], a_l1)
                    T.copy(B[0:TILE_N, kt * TILE_K : (kt + 1) * TILE_K], b_l1)
                    T.ascend_set_flag("MTE2_MTE1", 0)

                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    if kt == 0:
                        T.ascend_wait_flag("FIX_M", 0)
                    T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=(kt == 0))
                    T.ascend_set_flag("MTE1_MTE2", 0)

                T.ascend_set_flag("M_FIX", 0)
                T.ascend_wait_flag("M_FIX", 0)
                # Wait for both AIVs to be ready before writing buf_a.
                T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 4)  # AIV sid=0 ready
                T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 20)  # AIV sid=1 ready
                T.dual_copy(acc, buf_a)
                T.ascend_cross_core_set_flag(4, "PIPE_FIX", 6)  # buf_a ready -> AIV sid=0
                T.ascend_cross_core_set_flag(4, "PIPE_FIX", 22)  # buf_a ready -> AIV sid=1
                T.ascend_set_flag("FIX_M", 0)

                # Tile 1: A[128:256] @ B^T -> acc -> buf_b (L0C -> UB, M-split).
                for kt in T.Serial(K_TILES):
                    T.ascend_wait_flag("MTE1_MTE2", 0)
                    T.copy(A[TILE_M : 2 * TILE_M, kt * TILE_K : (kt + 1) * TILE_K], a_l1)
                    T.copy(B[0:TILE_N, kt * TILE_K : (kt + 1) * TILE_K], b_l1)
                    T.ascend_set_flag("MTE2_MTE1", 0)

                    T.ascend_wait_flag("MTE2_MTE1", 0)
                    if kt == 0:
                        T.ascend_wait_flag("FIX_M", 0)
                    T.gemm(a_l1, b_l1, acc, transpose_B=True, clear_accum=(kt == 0))
                    T.ascend_set_flag("MTE1_MTE2", 0)

                T.ascend_set_flag("M_FIX", 0)
                T.ascend_wait_flag("M_FIX", 0)
                # Wait for AIVs to finish reading buf_a (UB slot freed for buf_b).
                T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 4)  # AIV sid=0 done with buf_a
                T.ascend_cross_core_wait_flag(4, "PIPE_FIX", 20)  # AIV sid=1 done with buf_a
                T.dual_copy(acc, buf_b)
                T.ascend_cross_core_set_flag(4, "PIPE_FIX", 8)  # buf_b ready -> AIV sid=0
                T.ascend_cross_core_set_flag(4, "PIPE_FIX", 24)  # buf_b ready -> AIV sid=1
                T.ascend_set_flag("FIX_M", 0)

                # Drain pending intra-core flags.
                T.ascend_wait_flag("MTE1_MTE2", 0)
                T.ascend_wait_flag("FIX_M", 0)

            with T.Vector() as sid:
                my_buf_a_flag = 6 + sid * 16  # AIC -> AIV: buf_a ready
                my_buf_b_flag = 8 + sid * 16  # AIC -> AIV: buf_b ready
                my_ready_flag = 4 + sid * 16  # AIV -> AIC: ready / consumed

                # Initial: signal AIC that UB is free for buf_a.
                T.ascend_cross_core_set_flag(4, "PIPE_MTE3", my_ready_flag)

                T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", my_buf_a_flag)
                # AIV sid writes its half of tile 0 to C[sid*HALF_M : (sid+1)*HALF_M].
                T.copy(buf_a, C[sid * HALF_M : (sid + 1) * HALF_M, 0:TILE_N])
                # Signal AIC: buf_a consumed (UB slot may be reused for buf_b).
                T.ascend_cross_core_set_flag(4, "PIPE_MTE3", my_ready_flag)

                T.ascend_cross_core_wait_flag(4, "PIPE_MTE3", my_buf_b_flag)
                # Tile 1 lands at rows [TILE_M, 2*TILE_M).
                T.copy(buf_b, C[TILE_M + sid * HALF_M : TILE_M + (sid + 1) * HALF_M, 0:TILE_N])

    return main


def ref_program(a, b):
    """Reference: C = A @ B^T (B is (N, K) pre-transposed)."""
    return (a.float() @ b.float().T).to(torch.float32)


def run_example(target="ascend"):
    M, K, N = 256, 256, 128
    dtype = torch.float16
    device = torch.device("npu")

    print(f"Compiling gemm_ub_merge (M={M}, K={K}, N={N}, target={target})...")
    program = gemm_ub_merge(M, K, N)
    kernel = tilelang.compile(
        program,
        target=target,
        out_idx=-1,
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False},
    )
    print("Compilation succeeded!")

    print("\n--- Generated Ascend Source ---")
    print(kernel.get_kernel_source())

    a = torch.randn(M, K, dtype=dtype, device=device)
    b = torch.randn(N, K, dtype=dtype, device=device)

    print("\nRunning kernel on NPU...")
    c = kernel(a, b)
    torch.npu.synchronize()

    expected = ref_program(a, b)
    max_diff = torch.max(torch.abs(c - expected)).item()
    print(f"  {'PASS' if max_diff < 1e-2 else 'FAIL'}  max_diff={max_diff:.2e}")
    if max_diff >= 1e-2:
        raise AssertionError(f"Results mismatch! Max diff: {max_diff}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Ascend UB-merge GEMM example.")
    parser.add_argument("--target", choices=["ascend", "pto"], default="ascend")
    args = parser.parse_args()
    run_example(target=args.target)
