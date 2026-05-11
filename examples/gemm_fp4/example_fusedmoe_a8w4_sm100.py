"""Simplified A8W4 fused MoE shared expert on SM100/SM110.

This mirrors the SM120 demo shape, but uses TCGEN05/TMEM:
  gate = input(FP8) x W_gate(FP4)
  up   = input(FP8) x W_up(FP4)
  out  = silu(gate) * up

The down projection is intentionally omitted here, matching the existing SM120
diagnostic example and keeping the kernel focused on mixed FP8xFP4 TCGEN05 GEMM.
"""

import os
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


def unpack_fp4_to_float(packed_int8, rows, cols):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed_int8.device)
    flat = packed_int8.to(torch.uint8).reshape(rows, cols // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    unpacked = torch.stack([lo, hi], dim=-1).reshape(rows, cols).to(torch.int64)
    return lut[unpacked]


def fusedmoe_a8w4_sm100(num_tokens, d_hidden, d_expert, block_token=128, block_hidden=128, block_expert=64, threads=128, num_stages=1):
    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        Input: T.Tensor((num_tokens, d_hidden), "float8_e4m3fn"),
        W_gate: T.Tensor((d_expert, d_hidden), T.float4_e2m1fn),
        W_up: T.Tensor((d_expert, d_hidden), T.float4_e2m1fn),
        Output: T.Tensor((num_tokens, d_expert), "float32"),
    ):
        with T.Kernel(
            T.ceildiv(d_expert, block_expert),
            T.ceildiv(num_tokens, block_token),
            threads=threads,
        ) as (bx, by):
            input_shared = T.alloc_shared((block_token, block_hidden), "float8_e4m3fn")
            gate_shared = T.alloc_shared((block_expert, block_hidden), T.float4_e2m1fn)
            up_shared = T.alloc_shared((block_expert, block_hidden), T.float4_e2m1fn)

            gate_tmem = T.alloc_tmem([block_token, block_expert], "float32")
            up_tmem = T.alloc_tmem([block_token, block_expert], "float32")
            gate_mbar = T.alloc_barrier(1)
            up_mbar = T.alloc_barrier(1)

            gate_local = T.alloc_fragment((block_token, block_expert), "float32")
            up_local = T.alloc_fragment((block_token, block_expert), "float32")

            for k in T.Pipelined(T.ceildiv(d_hidden, block_hidden), num_stages=num_stages):
                T.copy(Input[by * block_token, k * block_hidden], input_shared)
                T.copy(W_gate[bx * block_expert, k * block_hidden], gate_shared)
                T.copy(W_up[bx * block_expert, k * block_hidden], up_shared)

                T.tcgen05_gemm(
                    input_shared,
                    gate_shared,
                    gate_tmem,
                    transpose_A=False,
                    transpose_B=True,
                    mbar=gate_mbar,
                    clear_accum=(k == 0),
                )
                T.mbarrier_wait_parity(gate_mbar, k % 2)

                T.tcgen05_gemm(
                    input_shared,
                    up_shared,
                    up_tmem,
                    transpose_A=False,
                    transpose_B=True,
                    mbar=up_mbar,
                    clear_accum=(k == 0),
                )
                T.mbarrier_wait_parity(up_mbar, k % 2)

            T.copy(gate_tmem, gate_local)
            T.copy(up_tmem, up_local)

            for i, j in T.Parallel(block_token, block_expert):
                gate = gate_local[i, j] * (1.0 / (1.0 + T.exp2(-gate_local[i, j] * scale)))
                up_local[i, j] = up_local[i, j] * gate

            T.copy(up_local, Output[by * block_token, bx * block_expert])

    return main


num_tokens = int(os.environ.get("TL_MOE_TOKENS", "128"))
d_hidden = int(os.environ.get("TL_MOE_HIDDEN", "256"))
d_expert = int(os.environ.get("TL_MOE_EXPERT", "256"))
block_token = int(os.environ.get("TL_MOE_BLOCK_TOKEN", "128"))
block_hidden = int(os.environ.get("TL_MOE_BLOCK_HIDDEN", "128"))
block_expert = int(os.environ.get("TL_MOE_BLOCK_EXPERT", "64"))

print(
    f"Running SM100 A8W4 fused MoE: tokens={num_tokens}, hidden={d_hidden}, "
    f"expert={d_expert}, block=({block_token},{block_hidden},{block_expert})"
)

func = fusedmoe_a8w4_sm100(num_tokens, d_hidden, d_expert, block_token, block_hidden, block_expert)
jit_kernel = tilelang.compile(
    func,
    out_idx=[3],
    target="cuda",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
print("Compilation succeeded!")

torch.manual_seed(42)
input_fp8 = torch.randn(num_tokens, d_hidden, device="cuda", dtype=torch.float16).to(torch.float8_e4m3fn)
w_gate = torch.randint(0, 256, (d_expert, d_hidden // 2), device="cuda", dtype=torch.uint8).to(torch.int8)
w_up = torch.randint(0, 256, (d_expert, d_hidden // 2), device="cuda", dtype=torch.uint8).to(torch.int8)

z_input = torch.zeros(num_tokens, d_hidden, device="cuda", dtype=torch.float8_e4m3fn)
z_gate = torch.zeros(d_expert, d_hidden // 2, device="cuda", dtype=torch.int8)
z_up = torch.zeros(d_expert, d_hidden // 2, device="cuda", dtype=torch.int8)
c_zero = jit_kernel(z_input, z_gate, z_up)
assert c_zero.abs().max().item() == 0.0, f"Zero test failed: max={c_zero.abs().max().item()}"
print("[PASS] zeros in -> zeros out")

out = jit_kernel(input_fp8, w_gate, w_up)
input_f32 = input_fp8.to(torch.float32)
gate_logits = input_f32 @ unpack_fp4_to_float(w_gate, d_expert, d_hidden).T
up_logits = input_f32 @ unpack_fp4_to_float(w_up, d_expert, d_hidden).T
ref = up_logits * (gate_logits * torch.sigmoid(gate_logits))

diff = (out.float() - ref).abs()
max_diff = diff.max().item()
rel_err = diff.sum().item() / (ref.abs().sum().item() + 1e-10)
print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
print("[PASS] MoE numerical verification" if rel_err < 0.05 else "[WARN] large diff")

torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    jit_kernel(input_fp8, w_gate, w_up)
torch.cuda.synchronize()
elapsed = (time.perf_counter() - start) / 100 * 1000
total_flops = 2 * num_tokens * d_hidden * d_expert * 2
print(f"Latency: {elapsed:.4f} ms")
print(f"TFLOPS:  {total_flops / (elapsed / 1e3) / 1e12:.2f}")
