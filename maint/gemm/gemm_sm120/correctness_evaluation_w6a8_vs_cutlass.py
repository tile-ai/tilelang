"""Compare TileLang's SM120 W6A8 (mxfp8 x mxfp6, kind::mxf8f6f4) vs CUTLASS.

Run from the repository root:

    python -m maint.gemm.gemm_sm120.correctness_evaluation_w6a8_vs_cutlass

Bands (both fp8 activation flavors where applicable):

- Controlled band (asserted bitwise): small-integer fp8 activations x
  full-range packed fp6 weights (fp6 fn-types have no Inf/NaN, all values
  dyadic) with a narrow power-of-two scale window - sums exact in fp32.
- Full code domain: random e5m2 activation bytes (Inf/NaN included); NaN
  masks must match and finite entries stay bitwise.
- Quantized band: bf16 -> mxfp8 (activations) + mxfp6 (weights) quantizer
  outputs through both engines, bitwise (the standard oracle for real data).

Set CUTLASS_ROOT to the CUTLASS checkout (official requirement v4.7.0).
"""

import os
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang.transform import simplify_prim_func


_SF_VEC_SIZE = 32
_WORD_SPAN = _SF_VEC_SIZE * 4
_TORCH_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_TILELANG_FP8 = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}


@simplify_prim_func
def _make_tilelang_w6a8_kernel(m: int, n: int, k: int, a_name: str):
    a_dtype = getattr(T, _TILELANG_FP8[a_name])
    accum_dtype = T.float32

    block_M = block_N = 128
    block_K = k
    threads = 128

    @T.prim_func
    def main(
        A: T.Tensor((m, k), a_dtype),
        B_blob: T.Tensor((n, k * 3 // 4), T.uint8),
        SFA: T.Tensor((m, k // _WORD_SPAN), T.uint32),
        SFB: T.Tensor((n, k // _WORD_SPAN), T.uint32),
        C: T.Tensor((m, n), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), a_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((block_N, block_K), T.float6_e3m2fn_unpacked, scope="shared.dyn")
            SFA_shared = T.alloc_shared((block_M, block_K // _WORD_SPAN), T.uint32, scope="shared.dyn")
            SFB_shared = T.alloc_shared((block_N, block_K // _WORD_SPAN), T.uint32, scope="shared.dyn")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            for ko in T.Pipelined((k // block_K), num_stages=1):
                for i, k_inner in T.Parallel(block_M, block_K):
                    A_shared[i, k_inner] = A[by * block_M + i, ko * block_K + k_inner]
                # SIMT producer of the b6x16_p32 padded smem form.
                for j, g, b in T.Parallel(block_N, block_K // 16, 12):
                    B_shared[j, 16 * g + b] = T.reinterpret(
                        T.float6_e3m2fn_unpacked, B_blob[bx * block_N + j, ko * (block_K * 3 // 4) + 12 * g + b]
                    )
                for i, k_inner in T.Parallel(block_M, block_K // _WORD_SPAN):
                    SFA_shared[i, k_inner] = SFA[by * block_M + i, ko * (block_K // _WORD_SPAN) + k_inner]
                for j, k_inner in T.Parallel(block_N, block_K // _WORD_SPAN):
                    SFB_shared[j, k_inner] = SFB[bx * block_N + j, ko * (block_K // _WORD_SPAN) + k_inner]
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    clear_accum=True,
                    k_start=0,
                    sf_a_granularity_k=_SF_VEC_SIZE,
                    sf_b_granularity_k=_SF_VEC_SIZE,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _pack_tilelang_sf_u32(sf_bytes):
    assert sf_bytes.dtype == torch.uint8
    mn, sf_blocks = sf_bytes.shape
    assert sf_blocks % 4 == 0
    words = sf_bytes.reshape(mn, sf_blocks // 4, 4).to(torch.int64)
    packed = words[:, :, 0] | (words[:, :, 1] << 8) | (words[:, :, 2] << 16) | (words[:, :, 3] << 24)
    return packed.to(torch.uint32).contiguous()


def _pack_cutlass_sf_bytes(sf_bytes):
    """Pack logical (MN, K/32) UE8M0 bytes into CUTLASS Sm1xxBlockScaledConfig<32> order."""
    assert sf_bytes.dtype == torch.uint8
    mn, sf_blocks = sf_bytes.shape
    assert mn % 128 == 0
    assert sf_blocks % 4 == 0

    rows = torch.arange(mn, device=sf_bytes.device).reshape(mn, 1)
    cols = torch.arange(sf_blocks, device=sf_bytes.device).reshape(1, sf_blocks)
    atom = rows // 128
    row_in_atom = rows % 128
    k_word_group = cols // 4
    byte_in_word = cols % 4
    offsets = atom * 128 * sf_blocks + k_word_group * 128 * 4 + (row_in_atom % 32) * 16 + (row_in_atom // 32) * 4 + byte_in_word
    out = torch.empty((mn * sf_blocks,), device=sf_bytes.device, dtype=torch.uint8)
    out[offsets.reshape(-1)] = sf_bytes.reshape(-1)
    return out.contiguous()


def _decode_fp6_blob(blob, cols):
    from examples.dequantize_gemm.quantize import decode_fp6_values, unpack_fp6_bytes

    return decode_fp6_values(unpack_fp6_bytes(blob.cpu(), cols), "e3m2").to(blob.device)


def _decode_ue8m0(sf_bytes):
    return torch.pow(2.0, (sf_bytes.to(torch.int32) - 127).to(torch.float32))


def _build_cutlass_extension():
    from torch.utils.cpp_extension import load

    source_path = Path(__file__).resolve().with_name("cutlass_w6a8_ref.cu")
    repo_root = source_path.parents[3]
    cutlass_root_env = os.environ.get("CUTLASS_ROOT")
    cutlass_root = Path(cutlass_root_env) if cutlass_root_env else repo_root / "3rdparty" / "cutlass"
    include_paths = [
        cutlass_root / "include",
        cutlass_root / "tools" / "util" / "include",
    ]
    if not all(path.exists() for path in include_paths):
        raise RuntimeError(
            "CUTLASS headers were not found. Set CUTLASS_ROOT to a CUTLASS checkout "
            "or populate the repository's 3rdparty/cutlass submodule."
        )

    return load(
        name="tilelang_cutlass_w6a8_ref",
        sources=[str(source_path)],
        extra_include_paths=[str(path) for path in include_paths],
        extra_cuda_cflags=[
            "-std=c++20",
            "-arch=sm_120a",
            "--expt-relaxed-constexpr",
            "-DCUTLASS_ARCH_MMA_SM120_SUPPORTED",
        ],
        extra_cflags=["-std=c++20"],
        verbose=True,
    )


def _make_varying_scales(mn, k):
    row = torch.arange(mn, device="cuda", dtype=torch.int32).reshape(mn, 1)
    col = torch.arange(k // _SF_VEC_SIZE, device="cuda", dtype=torch.int32).reshape(1, k // _SF_VEC_SIZE)
    return (127 + (row + col) % 3 - 1).to(torch.uint8)


def _run_pair(ext, kernel, a, b_blob, sfa_logical, sfb_logical, m, n, k, a_name):
    C_tl = kernel(a, b_blob, _pack_tilelang_sf_u32(sfa_logical), _pack_tilelang_sf_u32(sfb_logical))
    C_ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    D_cutlass = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    ext.cutlass_w6a8_gemm(
        a.view(torch.int8).contiguous(),
        b_blob.view(torch.int8).contiguous(),
        _pack_cutlass_sf_bytes(sfa_logical),
        _pack_cutlass_sf_bytes(sfb_logical),
        C_ref,
        D_cutlass,
        m,
        n,
        k,
        a_name == "e4m3",
    )
    return C_tl, D_cutlass


def _compare_one(ext, a_name, m, n, k, full_domain=False):
    b_blob = torch.randint(0, 256, (n, k * 3 // 4), device="cuda", dtype=torch.uint8)
    if full_domain:
        a = torch.randint(0, 256, (m, k), device="cuda", dtype=torch.uint8).view(_TORCH_FP8[a_name])
    else:
        a = torch.randint(-4, 5, (m, k), device="cuda", dtype=torch.float32).to(_TORCH_FP8[a_name])
    sfa_logical = _make_varying_scales(m, k)
    sfb_logical = _make_varying_scales(n, k)
    kernel = tilelang.compile(_make_tilelang_w6a8_kernel(m, n, k, a_name), target="cuda", out_idx=[4])
    C_tl, D_cutlass = _run_pair(ext, kernel, a, b_blob, sfa_logical, sfb_logical, m, n, k, a_name)

    band = ("full-domain" if full_domain else "controlled") + f" A={a_name}"
    if full_domain:
        nan_tl = torch.isnan(C_tl)
        assert torch.equal(nan_tl, torch.isnan(D_cutlass)), f"{band}: NaN masks differ"
        finite = ~nan_tl
        assert torch.equal(C_tl[finite].view(torch.int32), D_cutlass[finite].view(torch.int32)), f"{band}: not bitwise"
        print(f"w6a8 [{band}]: NaN masks equal, finite entries bitwise equal")
    else:
        assert torch.equal(C_tl, D_cutlass), f"{band}: not bitwise"
        sa = _decode_ue8m0(sfa_logical).repeat_interleave(_SF_VEC_SIZE, dim=1)
        sb = _decode_ue8m0(sfb_logical).repeat_interleave(_SF_VEC_SIZE, dim=1)
        ref = (a.to(torch.float32) * sa) @ (_decode_fp6_blob(b_blob, k) * sb).T
        print(f"w6a8 [{band}]: TileLang vs CUTLASS bitwise (python ref max diff {(C_tl - ref).abs().max().item():.3e})")


def _compare_quantized(ext, a_name, m, n, k):
    from examples.dequantize_gemm.quantize import (
        quantize_bf16_to_mxfp6_blockscaled,
        quantize_bf16_to_mxfp8_blockscaled,
    )

    x_a = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 2.0).to(torch.bfloat16)
    x_b = torch.randn(n, k, dtype=torch.float32) * 2.0
    a, _, sfa_logical = quantize_bf16_to_mxfp8_blockscaled(x_a, dtype=a_name, return_scale_bytes=True)
    b_blob, _, sfb_logical = quantize_bf16_to_mxfp6_blockscaled(x_b, dtype="e3m2", return_scale_bytes=True)
    kernel = tilelang.compile(_make_tilelang_w6a8_kernel(m, n, k, a_name), target="cuda", out_idx=[4])
    C_tl, D_cutlass = _run_pair(ext, kernel, a, b_blob.cuda(), sfa_logical, sfb_logical.cuda(), m, n, k, a_name)
    assert torch.equal(C_tl.view(torch.int32), D_cutlass.view(torch.int32)), f"quantized A={a_name} not bitwise"
    print(f"w6a8 [quantized bf16->mxfp8 x mxfp6, A={a_name}]: TileLang vs CUTLASS bitwise equal")


def run_compare() -> None:
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"
    ext = _build_cutlass_extension()

    m = n = 256
    k = 256
    for a_name in ("e4m3", "e5m2"):
        _compare_one(ext, a_name, m, n, k)
    _compare_one(ext, "e5m2", m, n, k, full_domain=True)
    _compare_quantized(ext, "e4m3", m, n, k)


if __name__ == "__main__":
    run_compare()
