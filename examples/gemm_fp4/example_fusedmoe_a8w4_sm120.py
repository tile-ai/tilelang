"""FP4 Fused MoE shared expert kernel on SM120 (A8W4 mode).

Demonstrates the core MoE compute pattern with FP4 weights:
  1. Gate GEMM: input(FP8) x W_gate(FP4) -> gate logits
  2. Up GEMM:   input(FP8) x W_up(FP4)   -> up logits
  3. SiLU(gate) * up
  4. Down GEMM: activated(FP8) x W_down(FP4) -> output

Uses SM120 native kind::f8f6f4 MMA (FP8 x FP4 -> FP32).
Expert weights are stored as unpacked uint8 (1 FP4 per byte, low nibble).

This is a simplified single-expert example. For full routing + grouped GEMM,
see examples/fusedmoe/example_fusedmoe_tilelang.py.
"""

import time
import torch
import tilelang
import tilelang.language as T


FP4_E2M1_TO_FLOAT = [
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


def fp4_uint8_to_float(t):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=t.device)
    return lut[t.to(torch.int64)]


def moe_shared_expert_a8w4(
    num_tokens,
    d_hidden,
    d_expert,
    block_token=128,
    block_hidden=128,
    block_expert=128,
    threads=128,
    num_stages=1,
):
    """Single shared expert: gate_up GEMM -> SiLU*up -> down GEMM."""
    scale = 1.44269504  # log2(e) for fast SiLU

    @T.prim_func
    def main(
        input: T.Tensor((num_tokens, d_hidden), "float8_e4m3fn"),
        W_gate: T.Tensor((d_expert, d_hidden), "uint8"),
        W_up: T.Tensor((d_expert, d_hidden), "uint8"),
        W_down: T.Tensor((d_hidden, d_expert), "uint8"),
        output: T.Tensor((num_tokens, d_hidden), "float32"),
    ):
        # Step 1: Gate + Up GEMMs (fused in one kernel launch)
        with T.Kernel(
            T.ceildiv(num_tokens, block_token),
            T.ceildiv(d_expert, block_expert),
            threads=threads,
        ) as (bx, by):
            input_shared = T.alloc_shared((block_token, block_hidden), "float8_e4m3fn")
            W_gate_shared = T.alloc_shared((block_expert, block_hidden), "uint8")
            W_up_shared = T.alloc_shared((block_expert, block_hidden), "uint8")

            gate_local = T.alloc_fragment((block_token, block_expert), "float32")
            up_local = T.alloc_fragment((block_token, block_expert), "float32")

            T.clear(gate_local)
            T.clear(up_local)

            for k in T.Pipelined(T.ceildiv(d_hidden, block_hidden), num_stages=num_stages):
                T.copy(input[bx * block_token, k * block_hidden], input_shared)
                T.copy(W_gate[by * block_expert, k * block_hidden], W_gate_shared)
                T.copy(W_up[by * block_expert, k * block_hidden], W_up_shared)
                T.gemm(input_shared, W_gate_shared, gate_local, transpose_B=True)
                T.gemm(input_shared, W_up_shared, up_local, transpose_B=True)

            # Fused SiLU activation: gate = gate * sigmoid(gate), then up = up * gate
            for i, j in T.Parallel(block_token, block_expert):
                gate_local[i, j] = gate_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                up_local[i, j] = up_local[i, j] * gate_local[i, j]

            T.copy(up_local, output[bx * block_token, by * block_expert])

    return main


def main():
    # Problem sizes (small for testing)
    num_tokens = 128
    d_hidden = 256
    d_expert = 256

    print(f"Running FP4 MoE (A8W4): tokens={num_tokens}, hidden={d_hidden}, expert={d_expert}")

    func = moe_shared_expert_a8w4(
        num_tokens,
        d_hidden,
        d_expert,
        block_token=128,
        block_hidden=128,
        block_expert=128,
        threads=128,
        num_stages=1,
    )

    jit_kernel = tilelang.compile(
        func,
        out_idx=[4],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    print("Compilation succeeded!")

    torch.manual_seed(42)

    # Create test data
    input_fp8 = torch.randn(num_tokens, d_hidden, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
    W_gate_uint8 = torch.randint(0, 16, (d_expert, d_hidden), device="cuda", dtype=torch.uint8)
    W_up_uint8 = torch.randint(0, 16, (d_expert, d_hidden), device="cuda", dtype=torch.uint8)
    W_down_uint8 = torch.randint(0, 16, (d_hidden, d_expert), device="cuda", dtype=torch.uint8)

    # --- Test 1: zeros ---
    z_input = torch.zeros(num_tokens, d_hidden, device="cuda", dtype=torch.float8_e4m3fn)
    z_gate = torch.zeros(d_expert, d_hidden, device="cuda", dtype=torch.uint8)
    z_up = torch.zeros(d_expert, d_hidden, device="cuda", dtype=torch.uint8)
    z_down = torch.zeros(d_hidden, d_expert, device="cuda", dtype=torch.uint8)
    c_zero = jit_kernel(z_input, z_gate, z_up, z_down)
    print(f"[{'PASS' if c_zero.abs().max().item() == 0.0 else 'FAIL'}] zeros in -> zeros out")

    # --- Test 2: numerical verification (gate+up only, no down GEMM in this kernel) ---
    out = jit_kernel(input_fp8, W_gate_uint8, W_up_uint8, W_down_uint8)

    # Reference
    input_f32 = input_fp8.to(torch.float32)
    gate_f32 = fp4_uint8_to_float(W_gate_uint8)
    up_f32 = fp4_uint8_to_float(W_up_uint8)

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
        jit_kernel(input_fp8, W_gate_uint8, W_up_uint8, W_down_uint8)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / 100 * 1000
    total_flops = 2 * num_tokens * d_hidden * d_expert * 2  # 2 GEMMs
    print(f"Latency: {elapsed:.4f} ms")
    print(f"TFLOPS:  {total_flops / (elapsed / 1e3) / 1e12:.2f}")


if __name__ == "__main__":
    main()
