"""Native NVFP4 fused MoE gate/up (FC1) example on SM100/SM110 (TCGEN05).

Matches the FlashInfer/TRT-LLM NVFP4 MoE compute paradigm (cf. flashinfer
`trtllm_fp4_block_scale_moe` / `cute_dsl_fused_moe_nvfp4`): activations AND expert
weights are NVFP4 (FP4 E2M1 data + FP8 E4M3 block scales, sf_vec_size=16); per
expert tile we run two block-scaled GEMMs (gate, up), then SiLU(gate) * up, then
apply routing. The down (FC2) projection and top-k routing are intentionally
omitted (single shared-expert demo), mirroring the SM120 NVFP4 MoE example.

This is the SM100/SM110 (DRIVE Thor) counterpart of example_fusedmoe_nvfp4_sm120.py:
it uses tcgen05.mma.kind::mxf4nvf4.block_scale with FP4 operands staged DENSE
(2 e2m1/byte) and E4M3 scale factors moved into TMEM. NOT yet hardware-verified.
"""

import argparse
import os
import time

import torch
import tilelang
import tilelang.language as T


FP4_E2M1_TO_FLOAT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def unpack_fp4_to_float(packed, rows, cols):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed.device)
    flat = packed.to(torch.uint8).reshape(rows, cols // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    codes = torch.stack([lo, hi], dim=-1).reshape(rows, cols).to(torch.int64)
    return lut[codes]


def fusedmoe_nvfp4_sm100(num_tokens, hidden, intermediate,
                         block_token=128, block_hidden=64, block_expert=128, threads=128):
    packed_hidden = hidden // 2
    packed_block_hidden = block_hidden // 2
    k_tiles = T.ceildiv(hidden, block_hidden)
    silu_scale = 1.44269504  # log2(e)

    @T.prim_func
    def main(
        X: T.Tensor((num_tokens, hidden), T.float4_e2m1fn),
        X_scale: T.Tensor((T.ceildiv(hidden, block_hidden) * num_tokens,), "uint32"),
        W_gate: T.Tensor((intermediate, hidden), T.float4_e2m1fn),
        W_gate_scale: T.Tensor((T.ceildiv(hidden, block_hidden) * intermediate,), "uint32"),
        W_up: T.Tensor((intermediate, hidden), T.float4_e2m1fn),
        W_up_scale: T.Tensor((T.ceildiv(hidden, block_hidden) * intermediate,), "uint32"),
        token_selected_experts: T.Tensor((num_tokens,), "int32"),
        token_final_scales: T.Tensor((num_tokens,), "float32"),
        Output: T.Tensor((num_tokens, intermediate), "float32"),
    ):
        with T.Kernel(T.ceildiv(intermediate, block_expert), T.ceildiv(num_tokens, block_token),
                      threads=threads) as (bx, by):
            # NVFP4 operands staged DENSE (two e2m1 per byte) as uint8.
            X_bytes = T.view(X, (num_tokens, packed_hidden), "uint8")
            Wg_bytes = T.view(W_gate, (intermediate, packed_hidden), "uint8")
            Wu_bytes = T.view(W_up, (intermediate, packed_hidden), "uint8")
            X_shared = T.alloc_shared((block_token, packed_block_hidden), "uint8")
            gate_shared = T.alloc_shared((block_expert, packed_block_hidden), "uint8")
            up_shared = T.alloc_shared((block_expert, packed_block_hidden), "uint8")
            sfx_shared = T.alloc_shared((block_token,), "uint32")
            sfg_shared = T.alloc_shared((block_expert,), "uint32")
            sfu_shared = T.alloc_shared((block_expert,), "uint32")

            gate_tmem = T.alloc_tmem([block_token, block_expert], "float32")
            up_tmem = T.alloc_tmem([block_token, block_expert], "float32")
            sfx_tmem = T.alloc_tmem([block_token, 4], "uint32")
            sfg_tmem = T.alloc_tmem([block_token, 4], "uint32")
            sfu_tmem = T.alloc_tmem([block_token, 4], "uint32")
            gate_mbar = T.alloc_barrier(1)
            up_mbar = T.alloc_barrier(1)

            gate_local = T.alloc_fragment((block_token, block_expert), "float32")
            up_local = T.alloc_fragment((block_token, block_expert), "float32")

            tx = T.get_thread_binding()
            T.use_swizzle(8)

            for ko in T.serial(k_tiles):
                T.copy(X_bytes[by * block_token, ko * packed_block_hidden], X_shared)
                T.copy(Wg_bytes[bx * block_expert, ko * packed_block_hidden], gate_shared)
                T.copy(Wu_bytes[bx * block_expert, ko * packed_block_hidden], up_shared)
                T.copy(X_scale[ko * num_tokens + by * block_token], sfx_shared)
                T.copy(W_gate_scale[ko * intermediate + bx * block_expert], sfg_shared)
                T.copy(W_up_scale[ko * intermediate + bx * block_expert], sfu_shared)
                T.sync_threads()

                if tx < 32:
                    T.tcgen05_sf_warp_transpose(sfx_shared)
                    T.tcgen05_sf_warp_transpose(sfg_shared)
                    T.tcgen05_sf_warp_transpose(sfu_shared)
                    T.fence_proxy_async()
                    T.tcgen05_cp_warpx4(sfx_shared, sfx_tmem)
                    T.tcgen05_cp_warpx4(sfg_shared, sfg_tmem)
                    T.tcgen05_cp_warpx4(sfu_shared, sfu_tmem)
                T.sync_threads()

                if 32 <= tx and tx < 64:
                    T.tcgen05_gemm_blockscaled(X_shared, gate_shared, gate_tmem, sfx_tmem, sfg_tmem,
                                               transpose_B=True, mbar=gate_mbar,
                                               clear_accum=(ko == 0), is_nvfp4=True)
                T.mbarrier_wait_parity(gate_mbar, ko % 2)
                if 32 <= tx and tx < 64:
                    T.tcgen05_gemm_blockscaled(X_shared, up_shared, up_tmem, sfx_tmem, sfu_tmem,
                                               transpose_B=True, mbar=up_mbar,
                                               clear_accum=(ko == 0), is_nvfp4=True)
                T.mbarrier_wait_parity(up_mbar, ko % 2)
                T.sync_threads()

            T.copy(gate_tmem, gate_local)
            T.copy(up_tmem, up_local)
            for t, i in T.Parallel(block_token, block_expert):
                token_idx = by * block_token + t
                gate = gate_local[t, i]
                silu = gate * (1.0 / (1.0 + T.exp2(-gate * silu_scale)))
                routed = T.if_then_else(token_selected_experts[token_idx] == 0,
                                        token_final_scales[token_idx], 0.0)
                Output[token_idx, bx * block_expert + i] = up_local[t, i] * silu * routed

    return main


def _decode_sf_full(sf, rows, hidden, block_hidden):
    """Decode flat group-major E4M3 scale words -> [rows, hidden] float scale."""
    sf_tiles = (hidden + block_hidden - 1) // block_hidden
    blocks_per_tile = block_hidden // 16  # 4 E4M3 per uint32
    raw = sf.view(torch.uint8).reshape(sf_tiles, rows, 4)[:, :, :blocks_per_tile].contiguous()
    sc = raw.view(torch.float8_e4m3fn).float()                 # [tile, row, blocks_per_tile]
    sc = sc.permute(1, 0, 2).reshape(rows, sf_tiles * blocks_per_tile)  # [row, hidden//16]
    return sc.repeat_interleave(16, dim=1)                     # [row, hidden]


def make_sf(rows, hidden, block_hidden):
    """Random per-(row, 16-block) E4M3 scale words, group-major flat layout."""
    sf_tiles = (hidden + block_hidden - 1) // block_hidden
    blocks_per_tile = block_hidden // 16
    n_blocks = sf_tiles * blocks_per_tile  # == hidden // 16
    vals = torch.rand(rows, n_blocks, device="cuda") * 1.75 + 0.25
    e4 = vals.to(torch.float8_e4m3fn).view(torch.uint8)
    out = torch.zeros(sf_tiles * rows, 4, device="cuda", dtype=torch.uint8)
    for t in range(sf_tiles):
        out[t * rows:(t + 1) * rows, :blocks_per_tile] = e4[:, t * blocks_per_tile:(t + 1) * blocks_per_tile]
    return out.contiguous().view(torch.uint32).reshape(sf_tiles * rows)


def main():
    parser = argparse.ArgumentParser(description="NVFP4 fused MoE (FC1) on SM100/SM110")
    parser.add_argument("-n", "--num-tokens", type=int, default=1024)
    parser.add_argument("-c", "--hidden-size", type=int, default=256)
    parser.add_argument("-s", "--intermediate-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("TL_MOE_WARMUP", "20")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("TL_MOE_ITERS", "100")))
    args = parser.parse_args()
    num_tokens, hidden, intermediate = args.num_tokens, args.hidden_size, args.intermediate_size
    block_hidden = 64

    print(f"Running SM100 NVFP4 fused MoE (kind::mxf4nvf4.block_scale): "
          f"tokens={num_tokens}, hidden={hidden}, intermediate={intermediate}")

    kernel = tilelang.compile(
        fusedmoe_nvfp4_sm100(num_tokens, hidden, intermediate, block_hidden=block_hidden),
        out_idx=[8], target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    print("Compilation succeeded!")

    torch.manual_seed(0)
    x = torch.randint(0, 256, (num_tokens, hidden // 2), device="cuda", dtype=torch.uint8)
    wg = torch.randint(0, 256, (intermediate, hidden // 2), device="cuda", dtype=torch.uint8)
    wu = torch.randint(0, 256, (intermediate, hidden // 2), device="cuda", dtype=torch.uint8)
    sfx = make_sf(num_tokens, hidden, block_hidden)
    sfg = make_sf(intermediate, hidden, block_hidden)
    sfu = make_sf(intermediate, hidden, block_hidden)
    selected = torch.zeros((num_tokens,), device="cuda", dtype=torch.int32)
    routing = torch.ones((num_tokens,), device="cuda", dtype=torch.float32)

    out = kernel(x, sfx, wg, sfg, wu, sfu, selected, routing)

    # Reference (flashinfer NVFP4 numerics): dequantize, gate/up GEMM, SiLU(gate)*up, route.
    sx = _decode_sf_full(sfx, num_tokens, hidden, block_hidden)
    sg = _decode_sf_full(sfg, intermediate, hidden, block_hidden)
    su = _decode_sf_full(sfu, intermediate, hidden, block_hidden)
    x_f = unpack_fp4_to_float(x, num_tokens, hidden) * sx
    gate = x_f @ (unpack_fp4_to_float(wg, intermediate, hidden) * sg).T
    up = x_f @ (unpack_fp4_to_float(wu, intermediate, hidden) * su).T
    routed = torch.where(selected == 0, routing, torch.zeros_like(routing)).unsqueeze(1)
    ref = up * (gate * torch.sigmoid(gate)) * routed

    diff = (out.float() - ref).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum().item() / (ref.abs().sum().item() + 1e-10)
    print(f"[NUMERICAL] max_abs_diff={max_diff:.4f}, rel_err={rel_err:.6f}")
    print("[PASS] MoE numerical verification" if rel_err < 0.05 else "[WARN] large diff (unverified)")

    for _ in range(args.warmup):
        kernel(x, sfx, wg, sfg, wu, sfu, selected, routing)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        kernel(x, sfx, wg, sfg, wu, sfu, selected, routing)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / max(args.iters, 1) * 1000
    total_flops = 2 * num_tokens * hidden * intermediate * 2  # gate + up
    print(f"Latency: {elapsed:.4f} ms")
    if elapsed > 0:
        print(f"TFLOPS:  {total_flops / (elapsed / 1e3) / 1e12:.2f}")


if __name__ == "__main__":
    main()
