"""Compare TileLang's SM120 MXFP8 (kind::mxf8f6f4) GEMM against CUTLASS.

Run from the repository root:

    python -m maint.gemm.gemm_sm120.correctness_evaluation_mxf8_vs_cutlass

All four {e4m3, e5m2} A/B pairings are compared. Two data bands:

- Controlled band (asserted bitwise): small-integer fp8 values and a narrow
  power-of-two scale window, so both engines perform the same exact fp32
  arithmetic in the same K order.
- Full code domain (e5m2 x e5m2 only): random raw bytes naturally include
  Inf/NaN; the NaN masks must match and the remaining entries stay bitwise.

Set CUTLASS_ROOT to the CUTLASS checkout to compile the reference against
(the official test requirement is v4.7.0); it defaults to the vendored
3rdparty/cutlass tree.
"""

import os
from pathlib import Path

import torch
import tilelang
import tilelang.language as T
from tilelang.transform import simplify_prim_func


_SF_VEC_SIZE = 32
_WORD_SPAN = _SF_VEC_SIZE * 4  # K elements covered by one uint32 scale word
_TORCH_FP8 = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}
_TILELANG_FP8 = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}


@simplify_prim_func
def _make_tilelang_mxf8_kernel(m: int, n: int, k: int, a_name: str, b_name: str, block_K: int | None = None):
    a_dtype = getattr(T, _TILELANG_FP8[a_name])
    b_dtype = getattr(T, _TILELANG_FP8[b_name])
    accum_dtype = T.float32

    block_M = block_N = 128
    block_K = k if block_K is None else block_K
    threads = 128

    @T.prim_func
    def main(
        A: T.Tensor((m, k), a_dtype),
        B: T.Tensor((n, k), b_dtype),
        SFA: T.Tensor((m, k // _WORD_SPAN), T.uint32),
        SFB: T.Tensor((n, k // _WORD_SPAN), T.uint32),
        C: T.Tensor((m, n), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(n, block_N), T.ceildiv(m, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), a_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((block_N, block_K), b_dtype, scope="shared.dyn")
            SFA_shared = T.alloc_shared((block_M, block_K // _WORD_SPAN), T.uint32, scope="shared.dyn")
            SFB_shared = T.alloc_shared((block_N, block_K // _WORD_SPAN), T.uint32, scope="shared.dyn")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            # clear_accum is a per-call clear; multi-ko loops hoist it.
            if k // block_K > 1:
                T.clear(C_local)
            for ko in T.Pipelined((k // block_K), num_stages=1):
                for i, k_inner in T.Parallel(block_M, block_K):
                    A_shared[i, k_inner] = A[by * block_M + i, ko * block_K + k_inner]
                for j, k_inner in T.Parallel(block_N, block_K):
                    B_shared[j, k_inner] = B[bx * block_N + j, ko * block_K + k_inner]
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
                    clear_accum=(k // block_K == 1),
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
    words_total = sf_blocks // 4
    offsets = atom * 128 * sf_blocks + k_word_group * 128 * 4 + (row_in_atom % 32) * 16 + (row_in_atom // 32) * 4 + byte_in_word
    out = torch.empty((mn * sf_blocks,), device=sf_bytes.device, dtype=torch.uint8)
    out[offsets.reshape(-1)] = sf_bytes.reshape(-1)
    assert words_total >= 1
    return out.contiguous()


def _decode_ue8m0(sf_bytes):
    return torch.pow(2.0, (sf_bytes.to(torch.int32) - 127).to(torch.float32))


def _python_reference(a, b, sfa_logical, sfb_logical):
    sfa = _decode_ue8m0(sfa_logical).repeat_interleave(_SF_VEC_SIZE, dim=1)
    sfb = _decode_ue8m0(sfb_logical).repeat_interleave(_SF_VEC_SIZE, dim=1)
    return (a.to(torch.float32) * sfa) @ (b.to(torch.float32) * sfb).T


def _build_cutlass_extension():
    from torch.utils.cpp_extension import load

    source_path = Path(__file__).resolve().with_name("cutlass_mxf8_ref.cu")
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
        name="tilelang_cutlass_mxf8_ref",
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


def _make_controlled_inputs(m, n, k, a_name, b_name):
    a = torch.randint(-4, 5, (m, k), device="cuda", dtype=torch.float32).to(_TORCH_FP8[a_name])
    b = torch.randint(-4, 5, (n, k), device="cuda", dtype=torch.float32).to(_TORCH_FP8[b_name])
    return a, b


def _make_varying_scales(mn, k):
    # Power-of-two exponents around 1.0, varying by row and K group; narrow
    # window so summation stays exact on top of the shared K order.
    row = torch.arange(mn, device="cuda", dtype=torch.int32).reshape(mn, 1)
    col = torch.arange(k // _SF_VEC_SIZE, device="cuda", dtype=torch.int32).reshape(1, k // _SF_VEC_SIZE)
    return (127 + (row + col) % 3 - 1).to(torch.uint8)


def _compare_one(ext, a_name, b_name, m, n, k, full_domain=False, block_K=None):
    if full_domain:
        a = torch.randint(0, 256, (m, k), device="cuda", dtype=torch.uint8).view(_TORCH_FP8[a_name])
        b = torch.randint(0, 256, (n, k), device="cuda", dtype=torch.uint8).view(_TORCH_FP8[b_name])
    else:
        a, b = _make_controlled_inputs(m, n, k, a_name, b_name)
    sfa_logical = _make_varying_scales(m, k)
    sfb_logical = _make_varying_scales(n, k)

    kernel = tilelang.compile(_make_tilelang_mxf8_kernel(m, n, k, a_name, b_name, block_K=block_K), target="cuda", out_idx=[4])
    C_tl = kernel(a, b, _pack_tilelang_sf_u32(sfa_logical), _pack_tilelang_sf_u32(sfb_logical))

    C_ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    D_cutlass = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    ext.cutlass_mxf8_gemm(
        a.view(torch.int8).contiguous(),
        b.view(torch.int8).contiguous(),
        _pack_cutlass_sf_bytes(sfa_logical),
        _pack_cutlass_sf_bytes(sfb_logical),
        C_ref,
        D_cutlass,
        m,
        n,
        k,
        a_name == "e4m3",
        b_name == "e4m3",
    )

    band = "full-domain" if full_domain else "controlled"
    band += f" bK={k if block_K is None else block_K}"
    if full_domain:
        nan_tl = torch.isnan(C_tl)
        nan_cutlass = torch.isnan(D_cutlass)
        assert torch.equal(nan_tl, nan_cutlass), f"{a_name}x{b_name} NaN masks differ"
        finite = ~nan_tl
        bitwise = torch.equal(C_tl[finite].view(torch.int32), D_cutlass[finite].view(torch.int32))
        assert bitwise, f"{a_name}x{b_name} {band}: finite entries not bitwise equal"
        print(f"{a_name}x{b_name} [{band}]: NaN masks equal, finite entries bitwise equal")
    else:
        max_abs = (C_tl - D_cutlass).abs().max().item()
        assert torch.equal(C_tl, D_cutlass), f"{a_name}x{b_name} {band}: max abs diff {max_abs}"
        ref = _python_reference(a, b, sfa_logical, sfb_logical)
        print(
            f"{a_name}x{b_name} [{band}]: TileLang vs CUTLASS bitwise equal "
            f"(both vs python reference max abs diff {(C_tl - ref).abs().max().item():.3e})"
        )


def _compare_quantized(ext, dtype_name, m, n, k):
    """The quantized band: real quantizer output, engine vs engine, bitwise.

    Feeding the SAME quantized tensors and scale bytes to both backends
    removes the python-reference tolerance problem entirely: both engines
    run the same instruction in the same K order, so the comparison is
    exact even though the data has mixed magnitudes and rounding partial
    sums.
    """
    from examples.dequantize_gemm.quantize import quantize_bf16_to_mxfp8_blockscaled

    x_a = (torch.randn(m, k, device="cuda", dtype=torch.float32) * 2.0).to(torch.bfloat16)
    x_b = (torch.randn(n, k, device="cuda", dtype=torch.float32) * 2.0).to(torch.bfloat16)
    a, _, sfa_logical = quantize_bf16_to_mxfp8_blockscaled(x_a, dtype=dtype_name, return_scale_bytes=True)
    b, _, sfb_logical = quantize_bf16_to_mxfp8_blockscaled(x_b, dtype=dtype_name, return_scale_bytes=True)

    kernel = tilelang.compile(_make_tilelang_mxf8_kernel(m, n, k, dtype_name, dtype_name), target="cuda", out_idx=[4])
    C_tl = kernel(a, b, _pack_tilelang_sf_u32(sfa_logical), _pack_tilelang_sf_u32(sfb_logical))

    C_ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    D_cutlass = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    ext.cutlass_mxf8_gemm(
        a.view(torch.int8).contiguous(),
        b.view(torch.int8).contiguous(),
        _pack_cutlass_sf_bytes(sfa_logical),
        _pack_cutlass_sf_bytes(sfb_logical),
        C_ref,
        D_cutlass,
        m,
        n,
        k,
        dtype_name == "e4m3",
        dtype_name == "e4m3",
    )
    assert torch.equal(C_tl.view(torch.int32), D_cutlass.view(torch.int32)), f"{dtype_name} quantized band not bitwise"
    print(f"{dtype_name}x{dtype_name} [quantized bf16->mxfp8]: TileLang vs CUTLASS bitwise equal")


def run_compare() -> None:
    torch.manual_seed(0)
    assert torch.cuda.is_available(), "CUDA is required"
    ext = _build_cutlass_extension()

    m = n = 256
    k = 256
    for a_name in ("e4m3", "e5m2"):
        for b_name in ("e4m3", "e5m2"):
            _compare_one(ext, a_name, b_name, m, n, k)
    # Full code domain for every pairing: e5m2 contributes Inf/NaN bytes,
    # e4m3 contributes its NaN encoding (0x7F); mixed pairs have no plain
    # T.gemm oracle, so CUTLASS is their only full-domain pin.
    for a_name in ("e4m3", "e5m2"):
        for b_name in ("e4m3", "e5m2"):
            _compare_one(ext, a_name, b_name, m, n, k, full_domain=True)
    # chunk=128: the config that was once silently broken (kblock4 fast-path
    # byte parity) stays pinned against CUTLASS forever.
    _compare_one(ext, "e4m3", "e4m3", m, n, k, block_K=128)
    _compare_one(ext, "e5m2", "e5m2", m, n, k, full_domain=True, block_K=128)
    # Real quantizer output through both engines, bitwise (no tolerance).
    _compare_quantized(ext, "e4m3", m, n, k)
    _compare_quantized(ext, "e5m2", m, n, k)


if __name__ == "__main__":
    run_compare()
