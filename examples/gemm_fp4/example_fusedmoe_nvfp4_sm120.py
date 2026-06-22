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


def make_nvfp4_sf(mn, K, block_mn, block_K=64):
    """Random NVFP4 E4M3 scale words for the SM120 layout: one word per
    (block_mn-row block, K64 tile); its 4 bytes = the 4 K16-group scales, shared
    across the block's rows. Returns (words [mn/block_mn, K/64] uint32, e4 [.., .., 4])."""
    sf_mn = (mn + block_mn - 1) // block_mn
    sf_k = (K + block_K - 1) // block_K
    groups = block_K // 16
    vals = torch.rand(sf_mn, sf_k, groups, device="cuda") * 1.75 + 0.25
    e4 = vals.to(torch.float8_e4m3fn).view(torch.uint8)
    words = e4.contiguous().view(torch.uint32).reshape(sf_mn, sf_k)
    return words, e4


def decode_nvfp4_sf(e4, mn, K, block_mn, block_K=64):
    """[sf_mn, sf_k, 4] E4M3 bytes -> [mn, K] float scale (byte b -> K16-group b,
    shared across block_mn rows)."""
    sc = e4.view(torch.float8_e4m3fn).float()
    sf_mn, sf_k, groups = sc.shape
    sc = sc.reshape(sf_mn, sf_k * groups)
    sc = sc.repeat_interleave(block_mn, dim=0)[:mn]
    return sc.repeat_interleave(16, dim=1)[:, :K]


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
    # Random NVFP4 scales: blocks are 16 rows (tokens) for X, 8 rows (experts) for W,
    # 64 along K (the m16n8k64 atom); one E4M3 word per (row-block, K64 tile).
    x_scale, e4x = make_nvfp4_sf(num_tokens, hidden_size, 16)
    w_gate_scale, e4g = make_nvfp4_sf(intermediate_size, hidden_size, 8)
    w_up_scale, e4u = make_nvfp4_sf(intermediate_size, hidden_size, 8)
    selected = torch.zeros((num_tokens,), device="cuda", dtype=torch.int32)
    routing = torch.ones((num_tokens,), device="cuda", dtype=torch.float32)

    out = kernel(x, x_scale, w_gate, w_gate_scale, w_up, w_up_scale, selected, routing)

    # Reference (FlashInfer NVFP4 numerics): dequantize with E4M3 scales, gate/up GEMM,
    # SiLU(gate)*up, then routing (expert 0 -> token_final_scales, else 0).
    sx = decode_nvfp4_sf(e4x, num_tokens, hidden_size, 16)
    sg = decode_nvfp4_sf(e4g, intermediate_size, hidden_size, 8)
    su = decode_nvfp4_sf(e4u, intermediate_size, hidden_size, 8)
    x_f32 = unpack_fp4_to_float(x, num_tokens, hidden_size) * sx
    gate = x_f32 @ (unpack_fp4_to_float(w_gate, intermediate_size, hidden_size) * sg).T
    up = x_f32 @ (unpack_fp4_to_float(w_up, intermediate_size, hidden_size) * su).T
    routed = torch.where(selected == 0, routing, torch.zeros_like(routing)).unsqueeze(1)
    ref = up * (gate * torch.sigmoid(gate)) * routed
    diff = (out.float() - ref).abs()
    rel_err = diff.sum().item() / (ref.abs().sum().item() + 1e-10)
    print(f"[NUMERICAL] max_abs_diff={diff.max().item():.4f}, rel_err={rel_err:.6f}")
    print("[PASS] numerical verification" if rel_err < 0.05 else "[WARN] large diff")

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
