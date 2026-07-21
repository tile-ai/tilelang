"""TileLang BF16 -> NVFP4 block-scaled quantizer kernel.

This is the performance quantizer used by the SM120 NVFP4 GEMM maint
benchmark. It builds on the device-side format helpers and layout contract in
``tilelang.quantize.nvfp4``; the library itself only ships those helpers plus
a torch reference implementation (``quantize_bf16_to_nvfp4_blockscaled``).
"""

import tilelang
import tilelang.language as T
from tilelang.quantize.nvfp4 import (
    _BLOCKSCALED_CHUNK_ROWS,
    _BLOCKSCALED_CHUNK_WORDS,
    _FP4_E2M1_MAX,
    _NVFP4_SCALE_BLOCK_K,
    _SCALE_BYTES_PER_WORD,
    _SCALE_LAYOUT_BLOCKSCALED_CHUNK_KMAJOR,
    _SCALE_LAYOUT_ID_BLOCKSCALED_CHUNK_KMAJOR,
    _import_torch,
    _scale_layout_id,
    _tl_encode_fp4_e2m1_code,
    _tl_encode_ue4m3_scale_byte_ceil,
    _tl_scale_source_coords,
    _tl_ue4m3_scale_byte_inverse,
)

_TILELANG_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
    tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
    tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
}


@tilelang.jit(pass_configs=_TILELANG_PASS_CONFIGS)
def _tilelang_nvfp4_blockscaled_quantize_kernel_tiled(
    x,
    rows_per_cta: int = 64,
    tma_store: bool = True,
    num_stages: int = 2,
    num_threads: int = 512,
    scale_layout_id: int = _SCALE_LAYOUT_ID_BLOCKSCALED_CHUNK_KMAJOR,
):
    """TileLang BF16 -> packed NVFP4 over a ``rows_per_cta x 256`` CTA tile.

    One kernel covers the tuned tile variants: TMA input staging, warp-strided
    row ownership (``rows_per_cta // num_warps`` rows per warp), and either a
    TMA store through shared memory (``tma_store=True``) or direct global
    stores. ``rows_per_cta`` in {16, 32, 64} reproduces the previous
    per-variant kernels exactly.
    """

    M = T.dynamic("M")
    K = T.const("K")
    rows_per_warp = rows_per_cta // (num_threads // 32)
    in_dtype = T.bfloat16
    compute_dtype = T.float32

    x: T.Tensor[(M, K), in_dtype]

    quant = T.empty((M, K // 2), T.int8)
    scale_source = T.empty((M, K // 64), T.uint32)

    with T.Kernel(M // rows_per_cta, K // 256, threads=num_threads) as (pid_m, pid_k256):
        lane = T.get_lane_idx()
        warp = T.get_warp_idx_sync()
        x_shared = T.alloc_shared((rows_per_cta, 256), in_dtype)
        if tma_store:
            quant_shared = T.alloc_shared((rows_per_cta, 128), T.int8)
        scale_codes = T.alloc_shared((rows_per_cta, 16), T.int32)

        T.copy(x[pid_m * rows_per_cta, pid_k256 * 256], x_shared, prefer_instruction="tma")
        T.sync_threads()

        for row_iter in T.serial(rows_per_warp):
            row = warp * rows_per_warp + row_iter
            for k64_inner in T.serial(_BLOCKSCALED_CHUNK_WORDS):
                elem_col = k64_inner * 64 + lane * 2
                byte_col = k64_inner * 32 + lane
                group = k64_inner * 4 + (lane // 8)
                group_lane = lane % 8
                value0 = T.cast(x_shared[row, elem_col], compute_dtype)
                value1 = T.cast(x_shared[row, elem_col + 1], compute_dtype)
                amax = T.alloc_var(compute_dtype, init=T.max(T.abs(value0), T.abs(value1)))
                amax = T.max(amax, T.shfl_xor(amax, 4, width=8))
                amax = T.max(amax, T.shfl_xor(amax, 2, width=8))
                amax = T.max(amax, T.shfl_xor(amax, 1, width=8))

                scale_code = T.alloc_var(T.int32, init=0)
                scale_inv = T.alloc_var(compute_dtype, init=0.0)
                if group_lane == 0:
                    scale_code = _tl_encode_ue4m3_scale_byte_ceil(amax / _FP4_E2M1_MAX)
                    scale_inv = _tl_ue4m3_scale_byte_inverse(scale_code)
                    scale_codes[row, group] = scale_code
                scale_inv = T.shfl_sync(scale_inv, 0, width=8)
                fp4_code0 = _tl_encode_fp4_e2m1_code(value0 * scale_inv)
                fp4_code1 = _tl_encode_fp4_e2m1_code(value1 * scale_inv)
                packed_byte = fp4_code0 | (fp4_code1 << 4)
                if tma_store:
                    quant_shared[row, byte_col] = T.cast(packed_byte, T.int8)
                else:
                    quant[pid_m * rows_per_cta + row, pid_k256 * 128 + byte_col] = T.cast(packed_byte, T.int8)

        T.sync_threads()
        for row_iter in T.serial(rows_per_warp):
            row = warp * rows_per_warp + row_iter
            if lane < _BLOCKSCALED_CHUNK_WORDS:
                k64_inner = lane
                scale_word = (
                    T.cast(scale_codes[row, k64_inner * 4], T.uint32)
                    | (T.cast(scale_codes[row, k64_inner * 4 + 1], T.uint32) << 8)
                    | (T.cast(scale_codes[row, k64_inner * 4 + 2], T.uint32) << 16)
                    | (T.cast(scale_codes[row, k64_inner * 4 + 3], T.uint32) << 24)
                )
                logical_row = pid_m * rows_per_cta + row
                k64_word = pid_k256 * _BLOCKSCALED_CHUNK_WORDS + k64_inner
                physical_row, physical_word = _tl_scale_source_coords(logical_row, k64_word, scale_layout_id, K // 64)
                scale_source[physical_row, physical_word] = scale_word

        if tma_store:
            T.sync_threads()
            T.tma_copy(quant_shared, quant[pid_m * rows_per_cta, pid_k256 * 128])
            T.tma_store_wait()

    return quant, scale_source


def tilelang_quantize_bf16_to_nvfp4_blockscaled(
    x,
    *,
    rows_per_cta: int = 32,
    num_stages: int = 4,
    num_threads: int = 512,
    scale_layout: str = _SCALE_LAYOUT_BLOCKSCALED_CHUNK_KMAJOR,
):
    """Quantize a CUDA BF16 activation tensor with a TileLang kernel.

    Parameters
    ----------
    x:
        CUDA ``torch.bfloat16`` tensor with shape ``[rows, K]``. ``rows`` must
        be a multiple of ``128`` and ``K`` a multiple of ``256``.
    rows_per_cta:
        CTA tile rows: ``16``, ``32`` or ``64`` (each CTA covers a
        ``rows_per_cta x 256`` tile).
    num_stages:
        Pipeline stage count for the TMA input staging.
    num_threads:
        CTA thread count; ``512`` (default) or ``1024`` for
        ``rows_per_cta`` in ``(32, 64)``.
    scale_layout:
        Scale-source memory layout. ``"blockscaled_chunk_kmajor"`` is the
        CUTLASS ``BlockScaledBasicChunk`` K-major source contract used by the
        TileLang SM120 NVFP4 GEMM path.

    Returns
    -------
    tuple
        ``(packed_fp4, scale_source)``. ``packed_fp4`` is ``torch.int8[rows, K/2]``.
        ``scale_source`` is ``torch.uint32[rows, K/64]`` in CUTLASS
        ``BlockScaledBasicChunk`` K-major source order.
        The same contract applies to SFA and SFB; ``rows`` is interpreted as M
        for A and N for B.
    """

    torch = _import_torch()
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x)!r}")
    if x.ndim != 2:
        raise ValueError(f"x must be a 2D tensor, got shape {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError("TileLang NVFP4 quantization requires a CUDA tensor")
    if x.dtype != torch.bfloat16:
        raise TypeError(f"TileLang NVFP4 quantization requires torch.bfloat16 input, got {x.dtype}")
    scale_layout_id = _scale_layout_id(scale_layout)
    if rows_per_cta not in (16, 32, 64):
        raise ValueError(f"rows_per_cta must be 16, 32, or 64, got {rows_per_cta}")
    if num_threads % 32 != 0:
        raise ValueError(f"num_threads must be a multiple of warp size 32, got {num_threads}")
    if rows_per_cta == 16 and num_threads != 512:
        raise ValueError("rows_per_cta=16 uses the one-warp-per-row schedule and requires num_threads=512")
    if rows_per_cta in (32, 64) and num_threads not in (512, 1024):
        raise ValueError("rows_per_cta=32/64 currently supports num_threads=512 or 1024")

    rows, cols = x.shape
    k_tile = _NVFP4_SCALE_BLOCK_K * _SCALE_BYTES_PER_WORD * _BLOCKSCALED_CHUNK_WORDS
    if rows % _BLOCKSCALED_CHUNK_ROWS != 0 or cols % k_tile != 0:
        raise ValueError(
            f"SM120 NVFP4 quantization requires rows multiple of {_BLOCKSCALED_CHUNK_ROWS} and K multiple of {k_tile}, got {tuple(x.shape)}"
        )

    quant, scale_source = _tilelang_nvfp4_blockscaled_quantize_kernel_tiled(
        x.contiguous(),
        rows_per_cta=rows_per_cta,
        tma_store=(rows_per_cta != 32),
        num_stages=num_stages,
        num_threads=num_threads,
        scale_layout_id=scale_layout_id,
    )
    return quant.contiguous().view(torch.int8), scale_source
