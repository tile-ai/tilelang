"""Regression: BF16 GEMM T.gemm numerical conformance on real Qwen shapes.

Historical context: the M0 canary initially flagged "BF16_GEMM_NUMERICAL_BUG"
(mean_rel up to 8.87 vs reference). DEFINITIVE re-test with correct weight
orientation and a CPU fp32-accumulation reference shows the flag was an
artifact of the test harness (weight not transposed to [K,N], and MPS fp32
matmul reference deviating ~1e-3 relative on large shapes), NOT a T.gemm
defect. T.gemm bf16 output is bit-identical to (a) the naive fp32-serial
kernel and (b) the MLX pack reference on real Qwen3-0.6B layer-0 q_proj.

This test pins that contract permanently: T.gemm bf16 must remain bit-identical
to the naive fp32-serial-accumulation reference and to the MLX pack values.
Any future T.gemm bf16 regression (accumulation, cast, tile, binding) fails
here with a clear diff.
"""

import numpy as np
import pytest
import torch

import tilelang
import tilelang.language as T

M, N, K = 304, 2048, 1024  # Qwen layer-0 q_proj padded shape
S = 290


@tilelang.jit
def gemm_tgemm(M, N, K, block_M=16, block_N=16, block_K=8):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((K, N), "bfloat16"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            As = T.alloc_shared((block_M, block_K), "bfloat16", scope="shared")
            Bs = T.alloc_shared((block_K, block_N), "bfloat16", scope="shared")
            Cl = T.alloc_shared((block_M, block_N), "float32", scope="shared")
            T.clear(Cl)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], As)
                T.copy(B[ko * block_K, bx * block_N], Bs)
                T.gemm(As, Bs, Cl)
            for i in T.serial(block_M):
                for j in T.serial(block_N):
                    C[by * block_M + i, bx * block_N + j] = T.cast(Cl[i, j], "bfloat16")

    return gemm


@tilelang.jit
def gemm_naive_fp32_serial(M, N, K):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), "bfloat16"),
        B: T.Tensor((K, N), "bfloat16"),
        C: T.Tensor((M, N), "bfloat16"),
    ):
        with T.Kernel(N, M) as (bx, by):
            acc = T.alloc_var(T.float32, 0.0)
            for k in T.serial(K):
                acc += T.cast(A[by, k], T.float32) * T.cast(B[k, bx], T.float32)
            C[by, bx] = T.cast(acc, "bfloat16")

    return gemm


@pytest.fixture(scope="module")
def qwen_inputs():
    import os
    from pathlib import Path

    root = Path(os.environ.get("M3_SANDBOX", "/Users/bgy/Downloads/M3-sandbox"))
    pack_path = root / "MAC_FORWARD_REFERENCE_V2" / "sample_000" / "torch.npz"
    if not pack_path.exists():
        pytest.skip("M0 workspace pack not available (set M3_SANDBOX)")
    pack = np.load(pack_path, allow_pickle=True)
    from safetensors.torch import load_file

    w = load_file(str(root / "vendor/stage1_models" / "Qwen3-0.6B-bf16" / "model.safetensors"))
    ln1 = pack["layer0.input_layernorm"][0]
    q_ref = pack["layer0.q_proj"][0]
    wq = w["model.layers.0.self_attn.q_proj.weight"].T.contiguous()
    return ln1, wq, q_ref


def test_tgemm_bf16_bitmatches_naive_and_pack(qwen_inputs):
    ln1, wq, q_ref = qwen_inputs
    ln1_pad = torch.zeros(M, K, dtype=torch.bfloat16, device="mps")
    ln1_pad[:S] = torch.from_numpy(ln1.astype(np.float32)).to(torch.bfloat16)
    wq_m = wq.to("mps")

    c_tg = torch.zeros(M, N, dtype=torch.bfloat16, device="mps")
    c_nv = torch.zeros(M, N, dtype=torch.bfloat16, device="mps")
    gemm_tgemm(M, N, K)(ln1_pad, wq_m, c_tg)
    gemm_naive_fp32_serial(M, N, K)(ln1_pad, wq_m, c_nv)
    torch.mps.synchronize()

    out_tg = c_tg[:S].float().cpu().numpy()
    out_nv = c_nv[:S].float().cpu().numpy()
    assert np.array_equal(out_tg, out_nv), f"T.gemm bf16 != naive fp32-serial: max_abs={np.abs(out_tg - out_nv).max()}"
    assert np.array_equal(out_tg, q_ref), f"T.gemm bf16 != MLX pack q_proj: max_abs={np.abs(out_tg - q_ref).max()}"
