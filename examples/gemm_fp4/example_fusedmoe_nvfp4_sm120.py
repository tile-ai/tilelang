"""Native NVFP4 fused MoE gate/up example on SM120.

This mirrors the FlashInfer/TRT-LLM NVFP4 data contract:

* activations and expert weights are declared as ``T.float4_e2m1fn`` tensors;
  the host still supplies torch uint8/int8 packed storage because PyTorch does
  not expose a scalar FP4 tensor dtype.
* scale factors are FP8 E4M3/UE4M3 values, packed four-at-a-time into the
  single scale register consumed by the current m16n8k64 MMA.SF wrapper.
* routing is represented by precomputed ``token_selected_experts`` and
  ``token_final_scales``.  This example implements the common fused FC1 pattern:
  gate/up GEMMs followed by SiLU(gate) * up.  The down projection is intentionally
  left to a future generic block-scaled GEMM tiling path.

The kernel is deliberately one CTA / one expert tile so the example validates the
native NVFP4 API and scale path without hiding behind uint8 tensor annotations.
"""

import os
import time

import torch
import tilelang
import tilelang.language as T
from tilelang.layout import make_swizzled_layout
import argparse

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


def pack_e4m3x4_to_u32(values: tuple[float, float, float, float]) -> int:
    raw = torch.tensor(values, dtype=torch.float32).to(torch.float8_e4m3fn).view(torch.uint8)
    return sum(int(byte) << (8 * i) for i, byte in enumerate(raw.tolist()))


def unpack_fp4_to_float(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed.device)
    flat = packed.to(torch.uint8).reshape(rows, cols // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(rows, cols).to(torch.int64)
    return lut[codes]


def fusedmoe_nvfp4_sm120(
    num_tokens=128,
    hidden_size=256,
    intermediate_size=128,
    block_tokens=16,
    block_hidden=64,
    block_intermediate=8,
    threads=32,
):
    assert num_tokens % block_tokens == 0
    assert hidden_size % block_hidden == 0
    assert intermediate_size % block_intermediate == 0

    packed_hidden = hidden_size // 2
    packed_block_hidden = block_hidden // 2
    token_blocks = num_tokens // block_tokens
    hidden_blocks = hidden_size // block_hidden
    intermediate_blocks = intermediate_size // block_intermediate
    scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        X: T.Tensor((num_tokens, hidden_size), T.float4_e2m1fn),
        X_scale: T.Tensor((token_blocks, hidden_blocks), "uint32"),
        W_gate: T.Tensor((intermediate_size, hidden_size), T.float4_e2m1fn),
        W_gate_scale: T.Tensor((intermediate_blocks, hidden_blocks), "uint32"),
        W_up: T.Tensor((intermediate_size, hidden_size), T.float4_e2m1fn),
        W_up_scale: T.Tensor((intermediate_blocks, hidden_blocks), "uint32"),
        token_selected_experts: T.Tensor((num_tokens,), "int32"),
        token_final_scales: T.Tensor((num_tokens,), "float32"),
        Output: T.Tensor((num_tokens, intermediate_size), "float32"),
    ):
        with T.Kernel(intermediate_blocks, token_blocks, threads=threads) as (bx, by):
            # Public tensors use native packed FP4.  The mxf4nvf4 ldmatrix path
            # consumes the raw packed bytes, so create byte views inside the
            # kernel instead of exposing uint8 tensors in the API.
            X_bytes = T.view(X, (num_tokens, packed_hidden), "uint8")
            W_gate_bytes = T.view(W_gate, (intermediate_size, packed_hidden), "uint8")
            W_up_bytes = T.view(W_up, (intermediate_size, packed_hidden), "uint8")
            X_shared = T.alloc_shared((block_tokens, packed_block_hidden), "uint8")
            gate_shared = T.alloc_shared((block_intermediate, packed_block_hidden), "uint8")
            up_shared = T.alloc_shared((block_intermediate, packed_block_hidden), "uint8")

            gate_local = T.alloc_fragment((block_tokens, block_intermediate), "float32")
            up_local = T.alloc_fragment((block_tokens, block_intermediate), "float32")
            SFA_local = T.alloc_local((1,), "uint32")
            SFB_local = T.alloc_local((1,), "uint32")

            T.annotate_layout(
                {
                    X_shared: make_swizzled_layout(X_shared),
                    gate_shared: make_swizzled_layout(gate_shared),
                    up_shared: make_swizzled_layout(up_shared),
                }
            )

            T.clear(gate_local)
            for ko in T.serial(hidden_blocks):
                T.copy(X_bytes[by * block_tokens, ko * packed_block_hidden], X_shared)
                T.copy(W_gate_bytes[bx * block_intermediate, ko * packed_block_hidden], gate_shared)
                SFA_local[0] = X_scale[by, ko]
                SFB_local[0] = W_gate_scale[bx, ko]
                T.nvfp4_gemm(X_shared, gate_shared, SFA_local, SFB_local, gate_local, transpose_B=True, clear_accum=(ko == 0))

            T.clear(up_local)
            for ko in T.serial(hidden_blocks):
                T.copy(X_bytes[by * block_tokens, ko * packed_block_hidden], X_shared)
                T.copy(W_up_bytes[bx * block_intermediate, ko * packed_block_hidden], up_shared)
                SFA_local[0] = X_scale[by, ko]
                SFB_local[0] = W_up_scale[bx, ko]
                T.nvfp4_gemm(X_shared, up_shared, SFA_local, SFB_local, up_local, transpose_B=True, clear_accum=(ko == 0))

            for t, i in T.Parallel(block_tokens, block_intermediate):
                gate = gate_local[t, i]
                up = up_local[t, i]
                token_idx = by * block_tokens + t
                out_idx = bx * block_intermediate + i
                routed = T.if_then_else(token_selected_experts[token_idx] == 0, token_final_scales[token_idx], 0.0)
                silu = gate * (1.0 / (1.0 + T.exp2(-gate * scale)))
                Output[token_idx, out_idx] = up * silu * routed

    return main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--num-tokens", type=int, default=1024)
    parser.add_argument("-c","--hidden-size", type=int, default=256)
    parser.add_argument("-s","--intermediate-size", type=int, default=1024)
    args = parser.parse_args()  
    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    one_scale = pack_e4m3x4_to_u32((1.0, 1.0, 1.0, 1.0))

    kernel = tilelang.compile(
        fusedmoe_nvfp4_sm120(num_tokens, hidden_size, intermediate_size),
        out_idx=[8],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    print("Compilation succeeded!")

    torch.manual_seed(0)
    x = torch.randint(0, 256, (num_tokens, hidden_size // 2), device="cuda", dtype=torch.uint8)
    w_gate = torch.randint(0, 256, (intermediate_size, hidden_size // 2), device="cuda", dtype=torch.uint8)
    w_up = torch.randint(0, 256, (intermediate_size, hidden_size // 2), device="cuda", dtype=torch.uint8)
    token_blocks = num_tokens // 16
    hidden_blocks = hidden_size // 64
    intermediate_blocks = intermediate_size // 8
    x_scale = torch.full((token_blocks, hidden_blocks), one_scale, device="cuda", dtype=torch.uint32)
    w_gate_scale = torch.full((intermediate_blocks, hidden_blocks), one_scale, device="cuda", dtype=torch.uint32)
    w_up_scale = torch.full((intermediate_blocks, hidden_blocks), one_scale, device="cuda", dtype=torch.uint32)
    selected = torch.zeros((num_tokens,), device="cuda", dtype=torch.int32)
    routing = torch.ones((num_tokens,), device="cuda", dtype=torch.float32)

    out = kernel(x, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, selected, routing)

    x_f32 = unpack_fp4_to_float(x, num_tokens, hidden_size)
    gate_f32 = unpack_fp4_to_float(w_gate, intermediate_size, hidden_size)
    up_f32 = unpack_fp4_to_float(w_up, intermediate_size, hidden_size)
    gate = x_f32 @ gate_f32.T
    up = x_f32 @ up_f32.T
    ref = up * (gate * torch.sigmoid(gate))
    diff = (out.float() - ref).abs()
    print(f"[NUMERICAL] max_abs_diff={diff.max().item():.4f}, rel_err={diff.sum().item() / (ref.abs().sum().item() + 1e-10):.6f}")

    warmup = int(os.environ.get("TL_NVFP4_MOE_WARMUP", "20"))
    iters = int(os.environ.get("TL_NVFP4_MOE_ITERS", "100"))
    for _ in range(warmup):
        kernel(x, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, selected, routing)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        kernel(x, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, selected, routing)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000 / iters
    # Two FC1 GEMMs: gate and up.  The SiLU/mul/routing epilogue is not included.
    total_flops = 2 * num_tokens * hidden_size * intermediate_size * 2
    tflops = total_flops / (elapsed_ms / 1e3) / 1e12
    print(f"[BENCH] latency={elapsed_ms:.4f} ms, TFLOPS={tflops:.2f}, iters={iters}")
