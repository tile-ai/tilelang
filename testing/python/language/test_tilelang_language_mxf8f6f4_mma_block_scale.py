"""SM120 ``kind::mxf8f6f4`` (MXFP8) block-scaled MMA tests.

Test data contract (two bands, both mandatory):

- Controlled band: small-integer fp8 values and a narrow power-of-two scale
  window, so every product and partial sum is exactly representable in fp32
  (K * dynamic-range << 2^24) and assertions run at ``atol=0``.
- Full code domain: random raw bytes, which for e5m2 naturally include
  Inf/NaN. Assertions on that band are masked (isnan/isinf handled
  separately); never shrink the data window to make an assertion pass.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.transform import simplify_prim_func


_FP8_TORCH_DTYPES = {"e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}


def _torch_fp8(name):
    import torch

    return {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[name]


@simplify_prim_func
def _make_mxf8_matmul_kernel(
    M,
    N,
    K,
    a_dtype,
    b_dtype,
    num_stages=1,
    *,
    block_M=64,
    block_N=64,
    block_K=128,
):
    accum_dtype = T.float32
    scale_words_per_block_k = block_K // 128

    @T.prim_func
    def main(
        A: T.Tensor((M, K), a_dtype),
        B: T.Tensor((N, K), b_dtype),
        SFA: T.Tensor((M, K // 128), T.uint32),
        SFB: T.Tensor((N, K // 128), T.uint32),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), a_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((block_N, block_K), b_dtype, scope="shared.dyn")
            SFA_shared = T.alloc_shared((block_M, scale_words_per_block_k), T.uint32, scope="shared.dyn")
            SFB_shared = T.alloc_shared((block_N, scale_words_per_block_k), T.uint32, scope="shared.dyn")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            if K // block_K > 1:
                T.clear(C_local)
            for ko in T.Pipelined((K // block_K), num_stages=num_stages):
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]
                for i, k in T.Parallel(block_M, scale_words_per_block_k):
                    SFA_shared[i, k] = SFA[by * block_M + i, ko * scale_words_per_block_k + k]
                for j, k in T.Parallel(block_N, scale_words_per_block_k):
                    SFB_shared[j, k] = SFB[bx * block_N + j, ko * scale_words_per_block_k + k]
                T.mma_gemm_blockscaled(
                    A_shared,
                    B_shared,
                    C_local,
                    SFA_shared,
                    SFB_shared,
                    transpose_B=True,
                    clear_accum=(K // block_K == 1),
                    # rowmajor scale addressing is relative to the staged
                    # per-ko slice (buffer-local).
                    k_start=0,
                    sf_a_granularity_k=32,
                    sf_b_granularity_k=32,
                )

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@simplify_prim_func
def _make_plain_fp8_matmul_kernel(M, N, K, a_dtype, b_dtype, *, block_M=64, block_N=64, block_K=128):
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), a_dtype),
        B: T.Tensor((N, K), b_dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), a_dtype, scope="shared.dyn")
            B_shared = T.alloc_shared((block_N, block_K), b_dtype, scope="shared.dyn")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined((K // block_K), num_stages=1):
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def _pack_scale_words(scale_bytes):
    import torch

    s = scale_bytes.to(torch.int64).reshape(scale_bytes.shape[0], -1, 4)
    w = s[:, :, 0] | (s[:, :, 1] << 8) | (s[:, :, 2] << 16) | (s[:, :, 3] << 24)
    return w.to(torch.uint32).contiguous()


def _make_controlled_fp8_inputs(M, N, K, a_name, b_name):
    """Small integers, exactly representable in both fp8 formats."""
    import torch

    A = torch.randint(-4, 5, (M, K), device="cuda", dtype=torch.float32).to(_torch_fp8(a_name))
    B = torch.randint(-4, 5, (N, K), device="cuda", dtype=torch.float32).to(_torch_fp8(b_name))
    return A, B


def _make_varying_scale_bytes(rows, K):
    """UE8M0 bytes from {0x7E, 0x7F, 0x80} = scales {0.5, 1, 2}."""
    import torch

    return (torch.randint(-1, 2, (rows, K // 32), device="cuda", dtype=torch.int64) + 127).to(torch.uint8)


def _reference_scaled_gemm(A, B, sfa_bytes, sfb_bytes):
    import torch

    sa = torch.pow(2.0, sfa_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    sb = torch.pow(2.0, sfb_bytes.to(torch.int32).float() - 127).repeat_interleave(32, dim=1)
    return (A.to(torch.float32) * sa) @ (B.to(torch.float32) * sb).T


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("a_name,b_name", [("e4m3", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e4m3"), ("e5m2", "e5m2")])
def test_mxf8f6f4_mma_block_scale_codegen(a_name, b_name):
    kernel = tilelang.compile(
        _make_mxf8_matmul_kernel(128, 128, 128, getattr(T, _FP8_TORCH_DTYPES[a_name]), getattr(T, _FP8_TORCH_DTYPES[b_name])),
        target="cuda",
        out_idx=[4],
    )
    src = kernel.get_kernel_source()
    assert "tl::SM120MmaBlockScaledKind::kMxf8f6f4" in src
    assert "tl::SM120MmaScaleType::kUE8M0" in src
    enum_of = {"e4m3": "kE4M3", "e5m2": "kE5M2"}
    assert f"tl::SM120MmaOperandType::{enum_of[a_name]}, tl::SM120MmaOperandType::{enum_of[b_name]}>" in src


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("a_name,b_name", [("e4m3", "e4m3"), ("e4m3", "e5m2"), ("e5m2", "e4m3"), ("e5m2", "e5m2")])
@pytest.mark.parametrize("K,block_K", [(128, 128), (256, 128), (512, 256)])
def test_mxf8f6f4_mma_block_scale_rowmajor_correctness(a_name, b_name, K, block_K):
    import torch

    torch.manual_seed(0)
    M = N = 128
    kernel = tilelang.compile(
        _make_mxf8_matmul_kernel(M, N, K, getattr(T, _FP8_TORCH_DTYPES[a_name]), getattr(T, _FP8_TORCH_DTYPES[b_name]), block_K=block_K),
        target="cuda",
        out_idx=[4],
    )
    A, B = _make_controlled_fp8_inputs(M, N, K, a_name, b_name)
    sfa_bytes = _make_varying_scale_bytes(M, K)
    sfb_bytes = _make_varying_scale_bytes(N, K)
    C = kernel(A, B, _pack_scale_words(sfa_bytes), _pack_scale_words(sfb_bytes))
    ref = _reference_scaled_gemm(A, B, sfa_bytes, sfb_bytes)
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
# Same-dtype pairs only: the generic T.gemm still rejects mixed fp8
# operands, so mixed pairs have no plain-MMA oracle; they are covered by the
# controlled-band python-reference tests above.
@pytest.mark.parametrize("a_name,b_name", [("e4m3", "e4m3"), ("e5m2", "e5m2")])
def test_mxf8f6f4_unit_scale_matches_plain_fp8_gemm_bitwise(a_name, b_name):
    """The unit-scale (SF = 0x7F = 1.0) fast-path oracle.

    kind::mxf8f6f4.block_scale with unit scales is numerically identical to
    the plain fp8 MMA the generic T.gemm emits, and both paths accumulate K
    in the same order, so the comparison is bitwise. Data is the full fp8
    code domain (random raw bytes; e5m2 includes Inf/NaN), compared through
    an integer view so NaN bit patterns participate.
    """
    import torch

    torch.manual_seed(0)
    M = N = 128
    K = 256
    a_dtype = getattr(T, _FP8_TORCH_DTYPES[a_name])
    b_dtype = getattr(T, _FP8_TORCH_DTYPES[b_name])
    scaled = tilelang.compile(_make_mxf8_matmul_kernel(M, N, K, a_dtype, b_dtype), target="cuda", out_idx=[4])
    plain = tilelang.compile(_make_plain_fp8_matmul_kernel(M, N, K, a_dtype, b_dtype), target="cuda", out_idx=[2])

    A = torch.randint(0, 256, (M, K), device="cuda", dtype=torch.uint8).view(_torch_fp8(a_name))
    B = torch.randint(0, 256, (N, K), device="cuda", dtype=torch.uint8).view(_torch_fp8(b_name))
    unit = torch.full((M, K // 128), 0x7F7F7F7F, device="cuda", dtype=torch.int64).to(torch.uint32)

    C_scaled = scaled(A, B, unit, torch.full((N, K // 128), 0x7F7F7F7F, device="cuda", dtype=torch.int64).to(torch.uint32))
    C_plain = plain(A, B)
    assert torch.equal(C_scaled.view(torch.int32), C_plain.view(torch.int32))


def _example_kmajor_kernel(M, N, K, block_K, in_dtype_name, num_stages=2):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "examples/gemm_sm120/sm120_mxfp8_blockscaled_gemm.py"
    spec = importlib.util.spec_from_file_location("sm120_mxfp8_blockscaled_gemm_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    kernel = module.sm120_mxfp8_blockscaled_gemm(M, N, K, 128, 128, block_K, num_stages, in_dtype_name, T.float32)
    return module, kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
@pytest.mark.parametrize("in_dtype_name", ["e4m3", "e5m2"])
def test_mxf8f6f4_kmajor_fulltile_correctness(in_dtype_name):
    """The performance path: blockscaled_chunk_kmajor fulltile, packed scales."""
    import torch

    torch.manual_seed(0)
    M = N = 256
    K = 512
    block_K = 128
    module, kernel = _example_kmajor_kernel(M, N, K, block_K, in_dtype_name)

    A = module._make_fp8(M, K, in_dtype_name, seed=0)
    B = module._make_fp8(N, K, in_dtype_name, seed=1)
    SFA_semantic = module._make_pow2_scale_words(M, K, seed=100)
    SFB_semantic = module._make_pow2_scale_words(N, K, seed=200)
    words = block_K // 128
    SFA = module.swizzle_blockscaled_chunk_kmajor_scale_words(SFA_semantic, block_words=words).reshape(-1, words)
    SFB = module.swizzle_blockscaled_chunk_kmajor_scale_words(SFB_semantic, block_words=words).reshape(-1, words)
    C = torch.empty((M, N), device="cuda", dtype=torch.float32)
    kernel(A, B, SFA, SFB, C)
    module._verify(A, B, SFA_semantic, SFB_semantic, C, torch.float32)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_mxf8f6f4_kmajor_scale_byte_literal_sequence():
    """Self-failing sentinel for the fulltile scale-byte stride.

    scale_vec::1X consumes one byte per k32 block, so the four unrolled
    kblocks of a K=128 word must select bytes exactly (0, 1, 2, 3). The
    pre-fix hardcoded ``* 2`` stride would produce (0, 2, 4, 6) and fail
    both assertions below.
    """
    import re

    _, kernel = _example_kmajor_kernel(128, 128, 128, 128, "e4m3")
    src = kernel.get_kernel_source()
    calls = re.findall(r"kMxf8f6f4[^;]+;", src)
    assert calls, "no mxf8f6f4 mma calls found in the generated source"
    for call in calls:
        args = re.findall(r"static_cast<uint16_t>\((.*?)\)[,)]", " ".join(call.split()))
        assert len(args) == 4, call  # byte_a, tid_a, byte_b, tid_b
        byte_expr = args[0]
        assert byte_expr == args[2]
        # The non-greedy capture can trim trailing parens; rebalance.
        byte_expr += ")" * (byte_expr.count("(") - byte_expr.count(")"))
        # The byte id is an expression in the kblock loop variable; evaluate
        # it over the four kblocks of one K=128 scale word.
        symbols = set(re.findall(r"[A-Za-z_]\w*", byte_expr))
        assert len(symbols) == 1, byte_expr
        (kvar,) = symbols
        seq = tuple(eval(byte_expr, {"__builtins__": {}}, {kvar: kb}) for kb in range(4))
        assert seq == (0, 1, 2, 3), seq
        assert max(seq) < 4


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_mxf8f6f4_e5m2_inf_nan_propagation():
    """Full-code-domain band: e5m2 Inf/NaN must propagate through the MMA."""
    import torch

    M = N = 128
    K = 128
    kernel = tilelang.compile(
        _make_mxf8_matmul_kernel(M, N, K, T.float8_e5m2, T.float8_e5m2),
        target="cuda",
        out_idx=[4],
    )
    A = torch.ones((M, K), device="cuda", dtype=torch.float32).to(torch.float8_e5m2)
    B = torch.ones((N, K), device="cuda", dtype=torch.float32).to(torch.float8_e5m2)
    A[0, 5] = float("inf")
    A[1, 7] = float("nan")
    unit = torch.full((M, K // 128), 0x7F7F7F7F, device="cuda", dtype=torch.int64).to(torch.uint32)
    C = kernel(A, B, unit, unit.clone())
    assert bool(torch.isinf(C[0]).all())  # inf * 1 accumulates to inf
    assert bool(torch.isnan(C[1]).all())  # nan poisons the whole row
    assert bool(torch.isfinite(C[2:]).all())


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_mxf8f6f4_scale_byte_0xff_yields_nan():
    """Recorded behavior: UE8M0 0xFF is the NaN scale encoding.

    The instruction propagates it as NaN into every product of the scaled
    block, poisoning the affected accumulator rows.
    """
    import torch

    M = N = 128
    K = 128
    kernel = tilelang.compile(
        _make_mxf8_matmul_kernel(M, N, K, T.float8_e4m3fn, T.float8_e4m3fn),
        target="cuda",
        out_idx=[4],
    )
    A, B = _make_controlled_fp8_inputs(M, N, K, "e4m3", "e4m3")
    sfa_bytes = torch.full((M, K // 32), 127, device="cuda", dtype=torch.uint8)
    sfb_bytes = sfa_bytes.clone()
    sfa_bytes[3, 1] = 0xFF
    C = kernel(A, B, _pack_scale_words(sfa_bytes), _pack_scale_words(sfb_bytes))
    assert bool(torch.isnan(C[3]).all())
    mask = torch.ones(M, dtype=torch.bool)
    mask[3] = False
    assert bool(torch.isfinite(C[mask]).all())


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_eq(12, 0)
def test_mxf8f6f4_many_tiles_many_stages_correctness():
    """Scale coverage: total_tiles (1024) > SM count, k stages > 2."""
    import torch

    torch.manual_seed(0)
    M = N = 2048
    K = 384
    kernel = tilelang.compile(
        _make_mxf8_matmul_kernel(M, N, K, T.float8_e4m3fn, T.float8_e4m3fn, num_stages=3, block_K=128),
        target="cuda",
        out_idx=[4],
    )
    A, B = _make_controlled_fp8_inputs(M, N, K, "e4m3", "e4m3")
    sfa_bytes = _make_varying_scale_bytes(M, K)
    sfb_bytes = _make_varying_scale_bytes(N, K)
    C = kernel(A, B, _pack_scale_words(sfa_bytes), _pack_scale_words(sfb_bytes))
    ref = _reference_scaled_gemm(A, B, sfa_bytes, sfb_bytes)
    torch.testing.assert_close(C, ref, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    tilelang.testing.main()
