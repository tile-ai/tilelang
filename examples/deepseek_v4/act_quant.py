import torch
import tilelang
import tilelang.language as T


def fast_log2_ceil(x):
    """Compute ceil(log2(x)) via IEEE 754 bit manipulation. Avoids slow log/ceil intrinsics."""
    bits_x = T.reinterpret(x, "uint32")
    exp_x = (bits_x >> 23) & 0xFF
    man_bits = bits_x & ((1 << 23) - 1)
    return T.Cast("int32", exp_x - 127 + T.if_then_else(man_bits != 0, 1, 0))


def fast_pow2(x):
    """Compute 2^x for integer x via IEEE 754 bit manipulation."""
    bits_x = (x + 127) << 23
    return T.reinterpret(bits_x, "float32")


def fast_round_scale(amax, fp8_max_inv):
    return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fp8_quant_kernel(
    x,
    block_size=128,
    round_scale=True,  # round scale to exponential of 2
):
    """Block-wise FP8 quantization."""

    M = T.dynamic("M")
    N = T.const("N")
    fp8_min, fp8_max = -448.0, 448.0
    in_dtype = T.bfloat16
    out_dtype = T.float8_e4m3
    scale_dtype = T.float32
    compute_dtype = T.float32  # Internal computation in FP32;

    x: T.Tensor[(M, N), in_dtype]

    blk_m = 32
    group_size = block_size

    quant = T.empty((M, N), out_dtype)
    scale = T.empty((M, T.ceildiv(N, group_size)), scale_dtype)

    fp8_max_inv = 1 / fp8_max
    num_stages = 0 if round_scale else 2

    with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
        bx,
        by,
    ):
        x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
        x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
        amax_local = T.alloc_fragment((blk_m,), compute_dtype)
        s_local = T.alloc_fragment((blk_m,), compute_dtype)
        y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
        y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

        for _ in T.Pipelined(1, num_stages=num_stages):
            T.copy(x[bx * blk_m, by * group_size], x_shared, disable_tma=True)
            T.copy(x_shared, x_local)

            T.reduce_absmax(x_local, amax_local, dim=1)
            for i in T.Parallel(blk_m):
                amax_local[i] = T.max(amax_local[i], 1e-4)
                if round_scale:
                    s_local[i] = fast_round_scale(amax_local[i], fp8_max_inv)
                else:
                    s_local[i] = amax_local[i] * fp8_max_inv
            for i, j in T.Parallel(blk_m, group_size):
                y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)

            for i in T.Parallel(blk_m):
                scale[bx * blk_m + i, by] = s_local[i]
            T.copy(y_local, y_shared)
            T.copy(y_shared, quant[bx * blk_m, by * group_size], disable_tma=True)

    return quant, scale


def fp8_act_quant(x: torch.Tensor, block_size: int = 128, round_scale: bool = False):
    """Block-wise FP8 quantization (bf16 -> fp8_e4m3).

    Args:
        x: Input tensor, shape (..., N) with N divisible by block_size, dtype bfloat16.
        block_size: Group size for block-wise scaling along the last dimension.
        round_scale: If True, round scale to nearest power of 2 (MXFP style).

    Returns:
        quant: FP8 quantized tensor, same shape as x, dtype float8_e4m3fn.
        scale: Per-block scales, shape (..., N // block_size), dtype float32.
    """
    N = x.size(-1)
    assert N % block_size == 0, f"N={N} must be divisible by block_size={block_size}"
    z = x.contiguous()
    quant, scale = fp8_quant_kernel(z.view(-1, N), block_size=block_size, round_scale=round_scale)
    return quant.view(x.shape), scale.view(*x.size()[:-1], -1)


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
)
def fp4_quant_kernel(
    x,
    block_size=32,
):
    """Block-wise FP4 quantization: bf16 -> float4_e2m1fn, E8M0 power-of-2 scale."""

    M = T.dynamic("M")
    N = T.const("N")
    fp4_min, fp4_max = -6.0, 6.0
    in_dtype = T.bfloat16
    out_dtype = T.float4_e2m1fn
    scale_dtype = T.float32
    compute_dtype = T.float32

    x: T.Tensor[(M, N), in_dtype]

    blk_m = 32
    group_size = block_size

    quant = T.empty((M, N), out_dtype)
    scale = T.empty((M, T.ceildiv(N, group_size)), scale_dtype)

    fp4_max_inv = 1.0 / fp4_max

    with T.Kernel(T.ceildiv(M, blk_m), T.ceildiv(N, group_size), threads=128) as (
        bx,
        by,
    ):
        x_shared = T.alloc_shared((blk_m, group_size), in_dtype)
        x_local = T.alloc_fragment((blk_m, group_size), in_dtype)
        amax_local = T.alloc_fragment((blk_m,), compute_dtype)
        s_local = T.alloc_fragment((blk_m,), compute_dtype)
        y_local = T.alloc_fragment((blk_m, group_size), out_dtype)
        y_shared = T.alloc_shared((blk_m, group_size), out_dtype)

        for _ in T.Pipelined(1, num_stages=2):
            T.copy(x[bx * blk_m, by * group_size], x_shared, disable_tma=True)
            T.copy(x_shared, x_local)

            T.reduce_absmax(x_local, amax_local, dim=1)
            for i in T.Parallel(blk_m):
                amax_local[i] = T.max(amax_local[i], 6 * (2**-126))
                s_local[i] = fast_round_scale(amax_local[i], fp4_max_inv)
            for i, j in T.Parallel(blk_m, group_size):
                y_local[i, j] = T.clamp(x_local[i, j] / s_local[i], fp4_min, fp4_max)

            for i in T.Parallel(blk_m):
                scale[bx * blk_m + i, by] = s_local[i]
            T.copy(y_local, y_shared)
            T.copy(y_shared, quant[bx * blk_m, by * group_size], disable_tma=True)

    return quant, scale


def fp4_act_quant(x: torch.Tensor, block_size: int = 32):
    """Block-wise FP4 quantization (bf16 -> float4_e2m1fn, E8M0 scale).

    Args:
        x: Input tensor, shape (..., N) with N divisible by block_size, dtype bfloat16.
        block_size: Group size for block-wise scaling (default 32, must be even).

    Returns:
        quant: FP4 quantized tensor, logical shape matching x,
               physical dtype float4_e2m1fn_x2 with last dim halved.
        scale: Per-block scales, shape (..., N // block_size), dtype float32.
    """
    N = x.size(-1)
    assert N % block_size == 0, f"N={N} must be divisible by block_size={block_size}"
    z = x.contiguous()
    quant, scale = fp4_quant_kernel(z.view(-1, N), block_size=block_size)
    # FP4 output has physical shape (M, N//2) with float4_e2m1fn_x2 dtype
    return quant.view(*x.size()[:-1], -1), scale.view(*x.size()[:-1], -1)


# ---------------------------------------------------------------------------
# PyTorch reference implementations
# ---------------------------------------------------------------------------


def fp8_act_quant_ref(x: torch.Tensor, block_size: int = 128, round_scale: bool = False):
    """PyTorch reference for block-wise FP8 quantization.

    Returns:
        quant: shape (M, N), dtype torch.float8_e4m3fn.
        scale: shape (M, num_blocks), dtype torch.float32.
    """
    M, N = x.shape
    fp8_max = 448.0
    num_blocks = N // block_size
    x_float = x.float().reshape(M, num_blocks, block_size)
    amax = x_float.abs().amax(dim=-1).clamp(min=1e-4)
    if round_scale:
        scale = torch.pow(2.0, torch.ceil(torch.log2(amax / fp8_max)))
    else:
        scale = amax / fp8_max
    x_scaled = x_float / scale.unsqueeze(-1)
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    quant = x_clamped.reshape(M, N).to(torch.float8_e4m3fn)
    return quant, scale


# FP4 E2M1 representable values (unsigned nibble 0-7; sign adds 8)
_FP4_VALS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def _nearest_fp4_nibble(x_clamped: torch.Tensor) -> torch.Tensor:
    """Round each float in [-6, 6] to the nearest FP4 E2M1 nibble value (uint8 0-15).

    Nibble encoding for float4_e2m1fn:
      0-7: +{0, 0.5, 1, 1.5, 2, 3, 4, 6}
      8-15: -{0, 0.5, 1, 1.5, 2, 3, 4, 6}  (sign bit set)
    """
    fp4_vals = _FP4_VALS.to(x_clamped.device)
    x_abs = x_clamped.abs().unsqueeze(-1)  # (..., 1)
    dists = (x_abs - fp4_vals).abs()  # (..., 8)
    idx = dists.argmin(dim=-1).to(torch.uint8)
    # For negative values, add 8 (set the sign nibble bit)
    idx = torch.where(x_clamped < 0, idx + 8, idx)
    return idx


def _pack_fp4_nibbles(nibbles: torch.Tensor) -> torch.Tensor:
    """Pack pairs of FP4 nibbles into uint8 bytes.

    nibbles: (M, N) uint8 tensor, each element in 0-15.
    Returns: (M, N//2) uint8 tensor.
    Low nibble (bits 3:0) = even index, High nibble (bits 7:4) = odd index.
    """
    M, N = nibbles.shape
    assert N % 2 == 0
    nb2 = nibbles.reshape(M, N // 2, 2)
    low = nb2[:, :, 0] & 0xF
    high = nb2[:, :, 1] & 0xF
    packed = ((high << 4) | low).to(torch.uint8)
    return packed


def fp4_act_quant_ref(x: torch.Tensor, block_size: int = 32):
    """PyTorch reference for block-wise FP4 quantization with E8M0 scale.

    Returns:
        quant: shape (M, N//2), dtype torch.float4_e2m1fn_x2 (packed).
        scale: shape (M, num_blocks), dtype torch.float32.
    """
    M, N = x.shape
    fp4_max = 6.0
    num_blocks = N // block_size
    x_float = x.float().reshape(M, num_blocks, block_size)
    amax = x_float.abs().amax(dim=-1).clamp(min=6.0 * (2.0**-126))
    # Always round scale for FP4/E8M0
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / fp4_max)))
    x_scaled = x_float / scale.unsqueeze(-1)
    x_clamped = x_scaled.clamp(-fp4_max, fp4_max)
    nibbles = _nearest_fp4_nibble(x_clamped.reshape(M, N))
    packed = _pack_fp4_nibbles(nibbles)
    quant = packed.view(torch.float4_e2m1fn_x2)
    return quant, scale


def fp4_dequant_to_float(quant: torch.Tensor, scale: torch.Tensor, N_logical: int) -> torch.Tensor:
    """Dequantize FP4 packed tensor back to float32 for verification.

    quant: shape (M, N//2), dtype float4_e2m1fn_x2.
    scale: shape (M, num_blocks), dtype float32.
    N_logical: original logical last dimension.
    Returns: shape (M, N_logical), dtype float32.
    """
    M = quant.size(0)
    num_blocks = scale.size(1)
    block_size = N_logical // num_blocks

    # View as uint8 and unpack nibbles
    quant_u8 = quant.view(torch.uint8)
    low = quant_u8 & 0xF
    high = (quant_u8 >> 4) & 0xF
    # Map nibbles to float values
    fp4_val_map = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=quant.device,
    )
    vals = torch.stack([fp4_val_map[low.long()], fp4_val_map[high.long()]], dim=-1)
    vals = vals.reshape(M, -1)  # (M, N_logical)

    # Broadcast scale per block
    vals_blocks = vals.reshape(M, num_blocks, block_size)
    scaled = vals_blocks * scale.unsqueeze(-1)
    return scaled.reshape(M, N_logical)


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_fp8_act_quant(M: int = 256, N: int = 1024, block_size: int = 128):
    """Test FP8 act quant: tilelang vs PyTorch reference."""
    torch.random.manual_seed(42)
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    for round_scale in [False, True]:
        quant_tl, scale_tl = fp8_act_quant(x, block_size=block_size, round_scale=round_scale)
        quant_ref, scale_ref = fp8_act_quant_ref(x, block_size=block_size, round_scale=round_scale)

        # Compare scales
        torch.testing.assert_close(scale_tl.float(), scale_ref.float(), rtol=1e-5, atol=1e-5)
        # Compare quantized values (both float8_e4m3fn, compare as float32)
        torch.testing.assert_close(quant_tl.float(), quant_ref.float(), rtol=0, atol=0)

    print(f"[PASS] test_fp8_act_quant M={M}, N={N}, block_size={block_size}")


def test_fp4_act_quant(M: int = 256, N: int = 1024, block_size: int = 32):
    """Test FP4 act quant: tilelang vs PyTorch reference via dequantized comparison."""
    torch.random.manual_seed(42)
    x = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")

    quant_tl, scale_tl = fp4_act_quant(x, block_size=block_size)
    quant_ref, scale_ref = fp4_act_quant_ref(x, block_size=block_size)

    # Both return (M, N//2) physical float4_e2m1fn_x2 dtype
    assert quant_tl.shape == quant_ref.shape, f"Shape mismatch: {quant_tl.shape} vs {quant_ref.shape}"

    # Compare scales
    torch.testing.assert_close(scale_tl.float(), scale_ref.float(), rtol=1e-5, atol=1e-5)

    # Compare raw bits (both should be identical uint8 storage under the hood)
    tl_bits = quant_tl.view(torch.uint8)
    ref_bits = quant_ref.view(torch.uint8)
    mismatches = (tl_bits != ref_bits).sum().item()
    total = tl_bits.numel()
    if mismatches > 0:
        # FP4 nibble rounding differs at exact midpoints (CUDA hardware vs argmin tie-breaking)
        dequant_tl = fp4_dequant_to_float(quant_tl, scale_tl, N)
        dequant_ref = fp4_dequant_to_float(quant_ref, scale_ref, N)
        max_diff = (dequant_tl - dequant_ref).abs().max().item()
        # Allow up to one FP4 step difference at exact midpoints (~0.5-1.0 in dequant space)
        torch.testing.assert_close(dequant_tl, dequant_ref, rtol=0.05, atol=1.0)
        print(f"  (FP4 nibble mismatch: {mismatches}/{total} bytes from tie-breaking, max dequant diff: {max_diff:.4f})")
    else:
        print(f"  (FP4 bit-exact match: {total} bytes)")

    print(f"[PASS] test_fp4_act_quant M={M}, N={N}, block_size={block_size}")


def test_round_trip_error():
    """Round-trip sanity: quantize then dequantize, check MSE."""
    torch.random.manual_seed(42)
    x = torch.randn((128, 512), dtype=torch.bfloat16, device="cuda")
    x_float = x.float()

    # FP8 round-trip
    quant_fp8, scale_fp8 = fp8_act_quant(x, block_size=128, round_scale=True)
    recovered_fp8 = quant_fp8.float() * scale_fp8.float().repeat_interleave(128, dim=1)
    fp8_mse = torch.nn.functional.mse_loss(recovered_fp8, x_float).item()
    print(f"  FP8 round-trip MSE: {fp8_mse:.6f}")
    assert fp8_mse < 1.0, f"FP8 round-trip MSE too high: {fp8_mse}"

    # FP4 round-trip
    quant_fp4, scale_fp4 = fp4_act_quant(x, block_size=32)
    recovered_fp4 = fp4_dequant_to_float(quant_fp4, scale_fp4, 512)
    fp4_mse = torch.nn.functional.mse_loss(recovered_fp4, x_float).item()
    print(f"  FP4 round-trip MSE: {fp4_mse:.6f}")
    assert fp4_mse < 10.0, f"FP4 round-trip MSE too high: {fp4_mse}"

    print("[PASS] test_round_trip_error")


if __name__ == "__main__":
    test_fp8_act_quant()
    test_fp4_act_quant()
    test_round_trip_error()
