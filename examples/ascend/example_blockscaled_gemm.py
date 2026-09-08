"""Ascend block-scaled GEMM example for MXFP8 and MXFP4."""

import argparse

import tilelang
import tilelang.language as T
from tilelang.profiler import do_bench


FP4_DTYPE = "float4_e2m1fn"
FP8_DTYPE = "float8_e4m3fn"


def gemm(
    M_DIM=8192,
    K_DIM=8192,
    N_DIM=8192,
    dtype="bfloat16",
    MIXED=None,
    hf32=None,
    scale_dtype="uint16",
    scale_packed=True,
    packed_fp4_input=False,
):
    """Build an L1 block-scaled GEMM for MXFP8 or MXFP4 operands."""

    NUM_BLOCKS = 32
    is_fp32 = dtype == "float32"
    is_fp4 = dtype == FP4_DTYPE
    if packed_fp4_input and not is_fp4:
        raise ValueError("packed_fp4_input is only valid for MXFP4")
    if is_fp4 and (scale_dtype != "uint16" or not scale_packed):
        raise ValueError("MXFP4 requires pair-packed uint16 scale factors")
    if MIXED is None:
        MIXED = not is_fp32

    TILE_M = 256
    TILE_N = 256
    TILE_K = 128 if is_fp32 else 512 if is_fp4 else 256

    SF_DIV = 64 if scale_dtype == "uint16" else 32
    SF_K = K_DIM // SF_DIV
    TILE_SF_K = TILE_K // SF_DIV
    SF_LOAD_CHUNK_SIZE = 1

    M_TILES = M_DIM // TILE_M
    N_TILES = N_DIM // TILE_N
    K_TILES = K_DIM // TILE_K
    OUT_TILES = M_TILES * N_TILES
    WINDOW = min(4, M_TILES)
    NUM_STAGES = 2
    MAIN_ROW = M_TILES // WINDOW - 1
    TAIL_WIN = M_TILES - MAIN_ROW * WINDOW
    x_shape = (M_DIM, K_DIM // 2) if packed_fp4_input else (M_DIM, K_DIM)
    w_shape = (N_DIM, K_DIM // 2) if packed_fp4_input else (N_DIM, K_DIM)
    input_dtype = "int8" if packed_fp4_input else dtype

    @T.macro
    def aswt_swizzle(tile_idx):
        m_tile = T.alloc_var("int32")
        n_tile = T.alloc_var("int32")
        row_idx = tile_idx // N_TILES // WINDOW
        if row_idx < MAIN_ROW:
            m_tile = row_idx * WINDOW + tile_idx % WINDOW
            n_tile = (tile_idx // WINDOW) % N_TILES
        else:
            tail_idx = tile_idx - MAIN_ROW * WINDOW * N_TILES
            m_tile = MAIN_ROW * WINDOW + tail_idx % TAIL_WIN
            n_tile = (tail_idx // TAIL_WIN) % N_TILES
        if row_idx % 2 != 0:
            n_tile = N_TILES - 1 - n_tile
        return m_tile, n_tile

    @T.prim_func
    def main(
        X: T.Buffer(x_shape, input_dtype),
        W: T.Buffer(w_shape, input_dtype),
        SFX: T.Buffer((SF_K, M_DIM) if scale_packed else (M_DIM, SF_K), scale_dtype),
        SFW: T.Buffer((SF_K, N_DIM) if scale_packed else (N_DIM, SF_K), scale_dtype),
        C: T.Buffer((M_DIM, N_DIM), "float32"),
    ):
        x_gm = T.view(X, (M_DIM, K_DIM), dtype) if packed_fp4_input else X
        w_gm = T.view(W, (N_DIM, K_DIM), dtype) if packed_fp4_input else W
        with T.Kernel(NUM_BLOCKS) as bx:
            if is_fp32:
                T.set_hf32_mode(hf32)
            res = T.alloc_l0c((TILE_M, TILE_N), "float32")
            x_l1 = T.alloc_l1((TILE_M, TILE_K), dtype)
            w_l1 = T.alloc_l1((TILE_N, TILE_K), dtype)
            xsf_l1 = T.alloc_l1((TILE_M, SF_LOAD_CHUNK_SIZE * TILE_SF_K), scale_dtype)
            wsf_l1 = T.alloc_l1((TILE_N, SF_LOAD_CHUNK_SIZE * TILE_SF_K), scale_dtype)
            temp = T.alloc_shared((TILE_M // 2, TILE_N), "float32")
            sf_int = SF_LOAD_CHUNK_SIZE

            for tile_idx in T.Persistent([OUT_TILES], NUM_BLOCKS, bx):
                m_tile, n_tile = aswt_swizzle(tile_idx)
                for kt in T.Pipelined(K_TILES, num_stages=NUM_STAGES):
                    T.copy(
                        x_gm[m_tile * TILE_M : (m_tile + 1) * TILE_M, kt * TILE_K : (kt + 1) * TILE_K],
                        x_l1,
                    )
                    T.copy(
                        w_gm[n_tile * TILE_N : (n_tile + 1) * TILE_N, kt * TILE_K : (kt + 1) * TILE_K],
                        w_l1,
                    )

                    if kt % sf_int == 0:
                        if scale_packed:
                            T.copy(
                                SFX[
                                    kt * TILE_SF_K : (kt + sf_int) * TILE_SF_K,
                                    m_tile * TILE_M : (m_tile + 1) * TILE_M,
                                ],
                                xsf_l1,
                                transpose=True,
                            )
                            T.copy(
                                SFW[
                                    kt * TILE_SF_K : (kt + sf_int) * TILE_SF_K,
                                    n_tile * TILE_N : (n_tile + 1) * TILE_N,
                                ],
                                wsf_l1,
                                transpose=True,
                            )
                        else:
                            T.copy(
                                SFX[
                                    m_tile * TILE_M : (m_tile + 1) * TILE_M,
                                    kt * TILE_SF_K : (kt + sf_int) * TILE_SF_K,
                                ],
                                xsf_l1,
                            )
                            T.copy(
                                SFW[
                                    n_tile * TILE_N : (n_tile + 1) * TILE_N,
                                    kt * TILE_SF_K : (kt + sf_int) * TILE_SF_K,
                                ],
                                wsf_l1,
                            )

                    sf_start = (kt % sf_int) * TILE_SF_K
                    T.blockscaled_gemm(
                        x_l1,
                        w_l1,
                        res,
                        sfa=xsf_l1[:, sf_start : sf_start + TILE_SF_K],
                        sfb=wsf_l1[:, sf_start : sf_start + TILE_SF_K],
                        transpose_B=True,
                        clear_accum=(kt == 0),
                        unit_flag_ctrl=T.Select(kt == K_TILES - 1, 3, 2),
                    )

                if MIXED:
                    T.dual_copy(res, temp, unit_flag_ctrl=3)
                    T.dual_copy(
                        temp,
                        C[
                            m_tile * TILE_M : (m_tile + 1) * TILE_M,
                            n_tile * TILE_N : (n_tile + 1) * TILE_N,
                        ],
                    )
                else:
                    T.copy(res, C[m_tile * TILE_M, n_tile * TILE_N], unit_flag_ctrl=3)

    return main


_FP4_VALS = None


def _fp4_vals(torch, device):
    global _FP4_VALS
    if _FP4_VALS is None or _FP4_VALS.device != device:
        _FP4_VALS = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32,
            device=device,
        )
    return _FP4_VALS


def _nearest_fp4_nibble(x_clamped):
    import torch

    fp4_vals = _fp4_vals(torch, x_clamped.device)
    index = (x_clamped.abs().unsqueeze(-1) - fp4_vals).abs().argmin(dim=-1).to(torch.uint8)
    return torch.where(x_clamped < 0, index + 8, index)


def _pack_fp4_nibbles(nibbles):
    rows, k = nibbles.shape
    assert k % 2 == 0
    pairs = nibbles.reshape(rows, k // 2, 2)
    return (((pairs[:, :, 1] & 0xF) << 4) | (pairs[:, :, 0] & 0xF)).to(nibbles.dtype)


def quantize_mxfp4_ref(x, block_size=32):
    """Quantize BF16/FP32 values into packed MXFP4 plus E8M0 scales."""

    import torch

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is None:
        raise RuntimeError("torch.float4_e2m1fn_x2 is required for MXFP4")

    rows, k = x.shape
    assert k % block_size == 0 and block_size % 2 == 0
    x_float = x.float().reshape(rows, k // block_size, block_size)
    fp4_max = 6.0
    amax = x_float.abs().amax(dim=-1).clamp(min=fp4_max * (2.0**-126))
    scale_exp = torch.ceil(torch.log2(amax / fp4_max)).to(torch.int16)
    scale = torch.pow(2.0, scale_exp.float()).to(x_float.dtype)
    scaled = (x_float / scale.unsqueeze(-1)).clamp(-fp4_max, fp4_max)
    packed = _pack_fp4_nibbles(_nearest_fp4_nibble(scaled.reshape(rows, k)))
    return packed.view(fp4_dtype), (scale_exp + 127).clamp(0, 255).to(torch.uint8)


def pack_e8m0_pairs_for_ascend(scale_e8m0):
    """Pack [rows, K/32] E8M0 bytes into Ascend [K/64, rows] uint16."""

    import torch

    assert scale_e8m0.shape[1] % 2 == 0
    return scale_e8m0.contiguous().view(torch.uint16).T.contiguous()


def dequant_mxfp4_ref(quant, scale_e8m0, logical_k):
    import torch

    quant_u8 = quant.view(torch.uint8)
    low = quant_u8 & 0xF
    high = (quant_u8 >> 4) & 0xF
    fp4_map = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=torch.float32,
        device=quant.device,
    )
    values = torch.stack([fp4_map[low.long()], fp4_map[high.long()]], dim=-1)
    values = values.reshape(quant.shape[0], logical_k)
    scales = torch.pow(2.0, scale_e8m0.float() - 127.0).repeat_interleave(32, dim=1)
    return values * scales[:, :logical_k]


def ref_program(x, w, sfx, sfw, logical_k=None):
    """Reference block-scaled matmul for MXFP8 or packed MXFP4 operands."""

    import torch

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is not None and x.dtype == fp4_dtype:
        logical_k = logical_k or x.shape[1] * 2
        xf = dequant_mxfp4_ref(x, sfx, logical_k)
        wf = dequant_mxfp4_ref(w, sfw, logical_k)
        return xf @ wf.T

    logical_k = logical_k or x.shape[1]
    sx = torch.pow(2.0, sfx.float() - 127.0).repeat_interleave(32, dim=1)
    sw = torch.pow(2.0, sfw.float() - 127.0).repeat_interleave(32, dim=1)
    return (x.float() * sx[:, :logical_k]) @ (w.float() * sw[:, :logical_k]).T


def _pack_scale_inputs(scale_e8m0, scale_dtype, scale_packed):
    import torch

    if scale_packed:
        if scale_dtype != "uint16":
            raise ValueError("pair-packed scales require scale_dtype='uint16'")
        return pack_e8m0_pairs_for_ascend(scale_e8m0)
    return scale_e8m0.contiguous().view(getattr(torch, scale_dtype))


def make_inputs(M, K, N, dtype, device, scale_dtype="uint16", scale_packed=True):
    """Create operands, physical scale tensors, and unpacked reference scales."""

    import torch

    if dtype == FP4_DTYPE:
        if scale_dtype != "uint16" or not scale_packed:
            raise ValueError("MXFP4 requires pair-packed uint16 scale factors")
        x_src = (torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        w_src = (torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.1).contiguous()
        x, sfx_e8m0 = quantize_mxfp4_ref(x_src)
        w, sfw_e8m0 = quantize_mxfp4_ref(w_src)
    else:
        x = (torch.randn(M, K, device=device) * 0.1).to(getattr(torch, dtype))
        w = (torch.randn(N, K, device=device) * 0.1).to(getattr(torch, dtype))
        sf_k_e8m0 = K // 32
        sfx_e8m0 = torch.randint(120, 135, (M, sf_k_e8m0), dtype=torch.uint8, device=device)
        sfw_e8m0 = torch.randint(120, 135, (N, sf_k_e8m0), dtype=torch.uint8, device=device)

    sfx = _pack_scale_inputs(sfx_e8m0, scale_dtype, scale_packed)
    sfw = _pack_scale_inputs(sfw_e8m0, scale_dtype, scale_packed)
    return x, w, sfx, sfw, sfx_e8m0, sfw_e8m0


def run_compile(
    M=8192,
    K=8192,
    N=8192,
    dtype=FP8_DTYPE,
    scale_dtype="uint16",
    scale_packed=True,
    print_source=False,
):
    program = gemm(M, K, N, dtype=dtype, scale_dtype=scale_dtype, scale_packed=scale_packed)
    kernel = tilelang.compile(program, out_idx=-1)
    if print_source:
        print(kernel.get_kernel_source())
    return kernel


def run_regression(
    M=8192,
    K=8192,
    N=8192,
    dtype=FP8_DTYPE,
    scale_dtype="uint16",
    scale_packed=True,
    print_source=False,
    bench=False,
):
    import torch

    device = torch.device("npu")
    kernel = run_compile(M, K, N, dtype, scale_dtype, scale_packed, print_source)
    x, w, sfx, sfw, sfx_e8m0, sfw_e8m0 = make_inputs(M, K, N, dtype, device, scale_dtype, scale_packed)
    result = kernel(x, w, sfx, sfw)
    torch.npu.synchronize()

    expected = ref_program(x, w, sfx_e8m0, sfw_e8m0, K)
    denom = expected.abs().mean().clamp_min(1e-6)
    rel_mean_diff = (result - expected).abs().mean().div(denom).item()
    print(f"{dtype}: rel_mean_diff={rel_mean_diff:.2e}")

    if bench:

        def run_kernel():
            return kernel(x, w, sfx, sfw)

        latency_ms = do_bench(run_kernel, backend="msprof", _n_warmup=30, _n_repeat=50)
        flops = 2.0 * M * N * K
        print(f"{latency_ms:.3f} ms/iter | {flops / (latency_ms / 1e3) / 1e12:.1f} TFLOPS")

    return rel_mean_diff


def run_regression_perf(M=8192, K=8192, N=8192, dtype=FP8_DTYPE):
    return run_regression(M, K, N, dtype=dtype, bench=True)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--k", type=int, default=8192)
    parser.add_argument("--n", type=int, default=8192)
    parser.add_argument("--dtype", choices=[FP8_DTYPE, FP4_DTYPE], default=FP8_DTYPE)
    parser.add_argument("--scale-dtype", choices=["uint8", "uint16"], default="uint16")
    parser.add_argument("--scale-unpacked", action="store_true")
    parser.add_argument("--print-source", action="store_true")
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    scale_packed = not args.scale_unpacked
    if args.compile_only:
        run_compile(
            args.m,
            args.k,
            args.n,
            args.dtype,
            args.scale_dtype,
            scale_packed,
            args.print_source,
        )
        print("Compilation succeeded")
    else:
        run_regression(
            args.m,
            args.k,
            args.n,
            args.dtype,
            args.scale_dtype,
            scale_packed,
            args.print_source,
            args.bench,
        )
