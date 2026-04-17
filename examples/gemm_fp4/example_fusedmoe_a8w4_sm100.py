"""FP4 Fused MoE shared expert kernel on SM100 (A8W4 mode).

Demonstrates the core MoE gate-up fusion with FP4 weights on SM100:
  1. Gate GEMM: input(FP8) x W_gate(FP4) -> gate logits  (TCGEN05MMA)
  2. Up GEMM:   input(FP8) x W_up(FP4)   -> up logits    (TCGEN05MMA)
  3. SiLU(gate) * up -> output

Uses SM100 TCGEN05MMA with TMEM accumulator and mixed-precision operands
(float8_e4m3fn x float4_e2m1fn -> float32).

Expert weights are stored as packed FP4 (2 values per byte, int8 storage).
This is a simplified single-expert example without routing.

Requires: SM100 GPU, CUDA 12.8+, PyTorch >= 2.4
"""

import time
import torch
import tilelang
import tilelang.language as T


FP4_E2M1_LUT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def pack_fp4_random(M: int, K: int, device="cuda") -> torch.Tensor:
    lo = torch.randint(0, 16, (M, K // 2), device=device, dtype=torch.uint8)
    hi = torch.randint(0, 16, (M, K // 2), device=device, dtype=torch.uint8)
    return ((hi << 4) | lo).to(torch.int8)


def unpack_fp4_to_float(packed: torch.Tensor, M: int, K: int) -> torch.Tensor:
    lut = torch.tensor(FP4_E2M1_LUT, dtype=torch.float32, device=packed.device)
    raw = packed.to(torch.uint8).reshape(M, K // 2)
    lo = raw & 0x0F
    hi = (raw >> 4) & 0x0F
    indices = torch.stack([lo, hi], dim=-1).reshape(M, K).to(torch.int64)
    return lut[indices]


def moe_shared_expert_a8w4_sm100(
    num_tokens,
    d_hidden,
    d_expert,
    block_token=128,
    block_hidden=128,
    block_expert=128,
    threads=256,
    num_stages=2,
):
    """Single shared expert: gate GEMM + up GEMM -> SiLU*up -> output."""
    scale = 1.44269504  # log2(e) for fast SiLU

    @T.prim_func
    def main(
        input: T.Tensor((num_tokens, d_hidden), "float8_e4m3fn"),
        W_gate: T.Tensor((d_expert, d_hidden), "float4_e2m1fn"),
        W_up: T.Tensor((d_expert, d_hidden), "float4_e2m1fn"),
        output: T.Tensor((num_tokens, d_expert), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(d_expert, block_expert),
            T.ceildiv(num_tokens, block_token),
            threads=threads,
        ) as (bx, by):
            input_shared = T.alloc_shared((block_token, block_hidden), "float8_e4m3fn")
            W_gate_shared = T.alloc_shared((block_expert, block_hidden), "float4_e2m1fn")
            W_up_shared = T.alloc_shared((block_expert, block_hidden), "float4_e2m1fn")

            gate_tmem = T.alloc_tmem([block_token, block_expert], "float32")
            up_tmem = T.alloc_tmem([block_token, block_expert], "float32")
            mbar_gate = T.alloc_barrier(1)
            mbar_up = T.alloc_barrier(1)

            gate_local = T.alloc_fragment((block_token, block_expert), "float32")
            up_local = T.alloc_fragment((block_token, block_expert), "float32")
            out_shared = T.alloc_shared((block_token, block_expert), "float32")

            k_iters = T.ceildiv(d_hidden, block_hidden)

            # Gate GEMM: input x W_gate^T
            for k in T.Pipelined(k_iters, num_stages=num_stages):
                T.copy(input[by * block_token, k * block_hidden], input_shared)
                T.copy(W_gate[bx * block_expert, k * block_hidden], W_gate_shared)
                T.gemm(
                    input_shared,
                    W_gate_shared,
                    gate_tmem,
                    False,
                    True,
                    mbar=mbar_gate,
                    wg_wait=-1,
                    clear_accum=k == 0,
                )
                T.mbarrier_wait_parity(mbar_gate, k % 2)

            T.copy(gate_tmem, gate_local)

            # Up GEMM: input x W_up^T
            for k in T.Pipelined(k_iters, num_stages=num_stages):
                T.copy(input[by * block_token, k * block_hidden], input_shared)
                T.copy(W_up[bx * block_expert, k * block_hidden], W_up_shared)
                T.gemm(
                    input_shared,
                    W_up_shared,
                    up_tmem,
                    False,
                    True,
                    mbar=mbar_up,
                    wg_wait=-1,
                    clear_accum=k == 0,
                )
                T.mbarrier_wait_parity(mbar_up, k % 2)

            T.copy(up_tmem, up_local)

            # Fused SiLU activation: output = up * (gate * sigmoid(gate))
            for i, j in T.Parallel(block_token, block_expert):
                gate_local[i, j] = gate_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                up_local[i, j] = up_local[i, j] * gate_local[i, j]

            T.copy(up_local, out_shared)
            T.copy(out_shared, output[by * block_token, bx * block_expert])

    return main


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    num_tokens = 128
    d_hidden = 256
    d_expert = 256

    block_token = 128
    block_hidden = 128
    block_expert = 128

    print(f"SM100 FusedMoE A8W4: tokens={num_tokens}, hidden={d_hidden}, expert={d_expert}")

    func = moe_shared_expert_a8w4_sm100(
        num_tokens,
        d_hidden,
        d_expert,
        block_token=block_token,
        block_hidden=block_hidden,
        block_expert=block_expert,
        threads=256,
        num_stages=2,
    )

    jit_kernel = tilelang.compile(
        func,
        out_idx=[3],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    print("Compilation succeeded!")

    torch.manual_seed(42)

    # Input: FP8 activation
    input_fp8 = torch.randn(num_tokens, d_hidden, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)

    # Weights: packed FP4 (2 per byte)
    W_gate_packed = pack_fp4_random(d_expert, d_hidden)
    W_up_packed = pack_fp4_random(d_expert, d_hidden)

    # --- Test 1: zeros ---
    z_input = torch.zeros(num_tokens, d_hidden, device="cuda", dtype=torch.float8_e4m3fn)
    z_gate = torch.zeros(d_expert, d_hidden // 2, device="cuda", dtype=torch.int8)
    z_up = torch.zeros(d_expert, d_hidden // 2, device="cuda", dtype=torch.int8)
    c_zero = jit_kernel(z_input, z_gate, z_up)
    status = "PASS" if c_zero.abs().max().item() == 0.0 else "FAIL"
    print(f"[{status}] zeros in -> zeros out")

    # --- Test 2: numerical verification ---
    out = jit_kernel(input_fp8, W_gate_packed, W_up_packed)

    # Reference
    input_f32 = input_fp8.to(torch.float32)
    gate_f32 = unpack_fp4_to_float(W_gate_packed, d_expert, d_hidden)
    up_f32 = unpack_fp4_to_float(W_up_packed, d_expert, d_hidden)

    gate_logits = input_f32 @ gate_f32.T
    up_logits = input_f32 @ up_f32.T
    gate_activated = gate_logits * torch.sigmoid(gate_logits)
    ref_out = up_logits * gate_activated

    diff = (out.float() - ref_out).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum().item() / (ref_out.abs().sum().item() + 1e-10)
    print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
    if rel_err < 0.05:
        print("[PASS] MoE gate+up fusion numerical verification")
    else:
        print("[WARN] large diff")

    # --- Benchmark ---
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(100):
        jit_kernel(input_fp8, W_gate_packed, W_up_packed)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 100 * 1000
    total_flops = 2 * num_tokens * d_hidden * d_expert * 2
    print(f"Latency: {elapsed:.4f} ms")
    print(f"TFLOPS:  {total_flops / (elapsed / 1e3) / 1e12:.2f}")
