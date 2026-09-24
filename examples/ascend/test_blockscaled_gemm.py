"""pytest tests for block-scaled GEMM with MXFP8 and MXFP4 E2M1."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T
from example_blockscaled_gemm import FP4_DTYPE, gemm, make_inputs, ref_program
from example_blockscaled_gemm_l0 import gemm as gemm_l0
from example_blockscaled_gemm_l0 import ref_program as ref_program_l0


def _make_sf_data(M, K, N, device):
    sf_k_e8m0 = K // 32
    sfx_e8m0 = torch.randint(120, 135, (M, sf_k_e8m0), dtype=torch.uint8, device=device)
    sfw_e8m0 = torch.randint(120, 135, (N, sf_k_e8m0), dtype=torch.uint8, device=device)
    sfx_pairs = sfx_e8m0.view(torch.uint16)
    sfw_pairs = sfw_e8m0.view(torch.uint16)
    sfx = sfx_pairs.T.contiguous()
    sfw = sfw_pairs.T.contiguous()
    return sfx, sfw, sfx_e8m0, sfw_e8m0


TARGETS = ["ascend"]


def _test(dtype, thresh, M=8192, K=8192, N=8192, seed=42, target=None):
    torch.manual_seed(seed)
    kernel = tilelang.compile(gemm(M, K, N, dtype=dtype), target=target, out_idx=-1)
    device = torch.device("npu")
    x = (torch.randn(M, K, device=device) * 0.1).to(getattr(torch, dtype))
    w = (torch.randn(N, K, device=device) * 0.1).to(getattr(torch, dtype))
    sfx, sfw, sfx_e8m0, sfw_e8m0 = _make_sf_data(M, K, N, device)
    c = kernel(x, w, sfx, sfw)
    torch.npu.synchronize()
    expected = ref_program(x.cpu(), w.cpu(), sfx_e8m0.cpu(), sfw_e8m0.cpu()).to(device)
    denom = expected.abs().mean().clamp_min(1e-6)
    rel_err = (c - expected).abs().mean() / denom
    assert rel_err.item() < thresh, f"{dtype} rel_err={rel_err.item():.2e}"
    return rel_err.item()


def _test_l0(dtype, thresh, M=4096, K=4096, N=4096, seed=42, target=None):
    torch.manual_seed(seed)
    kernel = tilelang.compile(gemm_l0(M, K, N, dtype=dtype), target=target, out_idx=-1)
    device = torch.device("npu")
    x = (torch.randn(M, K, device=device) * 0.1).to(getattr(torch, dtype))
    w = (torch.randn(N, K, device=device) * 0.1).to(getattr(torch, dtype))
    sfx, sfw, sfx_e8m0, sfw_e8m0 = _make_sf_data(M, K, N, device)
    c = kernel(x, w, sfx, sfw)
    torch.npu.synchronize()
    expected = ref_program_l0(x.cpu(), w.cpu(), sfx_e8m0.cpu(), sfw_e8m0.cpu()).to(device)
    denom = expected.abs().mean().clamp_min(1e-6)
    rel_err = (c - expected).abs().mean() / denom
    assert rel_err.item() < thresh, f"{dtype} L0 rel_err={rel_err.item():.2e}"
    return rel_err.item()


def _test_fp4(gemm_impl, path, thresh=1e-5, M=256, K=512, N=256, seed=42, target=None):
    torch.manual_seed(seed)
    kernel = tilelang.compile(
        gemm_impl(M, K, N, dtype=FP4_DTYPE),
        target=target,
        out_idx=-1,
    )
    device = torch.device("npu")
    x, w, sfx, sfw, sfx_e8m0, sfw_e8m0 = make_inputs(M, K, N, FP4_DTYPE, device)
    actual = kernel(x, w, sfx, sfw)
    torch.npu.synchronize()
    expected = ref_program(
        x.cpu(),
        w.cpu(),
        sfx_e8m0.cpu(),
        sfw_e8m0.cpu(),
        logical_k=K,
    ).to(device)
    denom = expected.abs().mean().clamp_min(1e-6)
    rel_err = (actual - expected).abs().mean() / denom
    assert rel_err.item() < thresh, f"FP4 {path} E2M1 rel_err={rel_err.item():.2e}"
    return rel_err.item()


def _test_fp4_byte_view(thresh=1e-5, M=512, K=1024, N=512, seed=42, target=None):
    torch.manual_seed(seed)
    program = gemm(M, K, N, dtype=FP4_DTYPE, packed_fp4_input=True)
    kernel = tilelang.compile(program, target=target, out_idx=-1)
    device = torch.device("npu")
    x, w, sfx, sfw, sfx_e8m0, sfw_e8m0 = make_inputs(M, K, N, FP4_DTYPE, device)
    actual = kernel(x.view(torch.int8), w.view(torch.int8), sfx, sfw)
    torch.npu.synchronize()
    expected = ref_program(
        x.cpu(),
        w.cpu(),
        sfx_e8m0.cpu(),
        sfw_e8m0.cpu(),
        logical_k=K,
    ).to(device)
    denom = expected.abs().mean().clamp_min(1e-6)
    rel_err = (actual - expected).abs().mean() / denom
    assert rel_err.item() < thresh, f"FP4 E2M1 byte-view rel_err={rel_err.item():.2e}"
    return rel_err.item()


def _make_l0_matrix_sf_transpose_kernel(M, K, N):
    dtype = "float8_e4m3fn"
    scale_dtype = "uint16"
    sf_k = K // 64

    @T.prim_func
    def gemm_kernel(
        X: T.Buffer((M, K), dtype),
        W: T.Buffer((N, K), dtype),
        SFX: T.Buffer((sf_k, M), scale_dtype),
        SFW: T.Buffer((sf_k, N), scale_dtype),
        C: T.Buffer((M, N), "float32"),
    ):
        with T.Kernel(1):
            res = T.alloc_l0c((M, N), "float32")
            x_l1 = T.alloc_l1((M, K), dtype)
            w_l1 = T.alloc_l1((N, K), dtype)
            xsf_l1 = T.alloc_l1((M, sf_k), scale_dtype)
            wsf_l1 = T.alloc_l1((N, sf_k), scale_dtype)
            x_l0a = T.alloc_l0a((M, K), dtype)
            w_l0b = T.alloc_l0b((N, K), dtype)
            x_l0a_sf = T.alloc_l0a_sf(x_l0a)
            w_l0b_sf = T.alloc_l0b_sf(w_l0b)

            T.copy(X, x_l1)
            T.copy(W, w_l1)
            T.copy(SFX, xsf_l1, transpose=True)
            T.copy(SFW, wsf_l1, transpose=True)
            T.copy(x_l1, x_l0a)
            T.copy(xsf_l1, x_l0a_sf)
            T.copy(w_l1, w_l0b)
            T.copy(wsf_l1, w_l0b_sf)
            T.gemm_blockscaled(x_l0a, w_l0b, res, x_l0a_sf, w_l0b_sf, transpose_B=True, clear_accum=True)
            T.copy(res, C)

    return gemm_kernel


def _test_l0_matrix_sf_transpose(M, K, N, thresh=2e-1, seed=42, target=None):
    torch.manual_seed(seed)
    kernel = tilelang.compile(
        _make_l0_matrix_sf_transpose_kernel(M, K, N),
        target=target,
        out_idx=-1,
    )
    device = torch.device("npu")
    x = (torch.randn(M, K, device=device) * 0.1).to(torch.float8_e4m3fn)
    w = (torch.randn(N, K, device=device) * 0.1).to(torch.float8_e4m3fn)
    sfx, sfw, sfx_e8m0, sfw_e8m0 = _make_sf_data(M, K, N, device)
    actual = kernel(x, w, sfx, sfw)
    torch.npu.synchronize()
    expected = ref_program_l0(x.cpu(), w.cpu(), sfx_e8m0.cpu(), sfw_e8m0.cpu()).to(device)
    denom = expected.abs().mean().clamp_min(1e-6)
    rel_err = (actual - expected).abs().mean() / denom
    assert rel_err.item() < thresh, f"L0 SF transpose shape=({M}, {K}, {N}) rel_err={rel_err.item():.2e}"
    return rel_err.item()


@pytest.mark.parametrize("target", TARGETS)
def test_blockscaled_gemm_l0_fp8_8192(target):
    err = _test_l0(
        "float8_e4m3fn",
        1e-2,
        M=8192,
        N=8192,
        K=8192,
        seed=123,
        target=target,
    )
    print(f"test_blockscaled_gemm_l0_fp8_8192[{target}]: PASS  rel_err={err:.2e}")


@pytest.mark.parametrize("target", TARGETS)
def test_blockscaled_gemm_l0_fp8_4096(target):
    err = _test_l0(
        "float8_e4m3fn",
        1e-2,
        M=4096,
        N=4096,
        K=4096,
        seed=456,
        target=target,
    )
    print(f"test_blockscaled_gemm_l0_fp8_4096[{target}]: PASS  rel_err={err:.2e}")


@pytest.mark.parametrize("target", TARGETS)
def test_blockscaled_gemm_fp8_8192(target):
    err = _test("float8_e4m3fn", 1e-2, target=target)
    print(f"test_blockscaled_gemm_fp8_8192[{target}]: PASS  rel_err={err:.2e}")


@pytest.mark.parametrize("target", TARGETS)
def test_blockscaled_gemm_fp8_4096(target):
    err = _test(
        "float8_e4m3fn",
        1e-2,
        M=4096,
        N=4096,
        K=4096,
        seed=123,
        target=target,
    )
    print(f"test_blockscaled_gemm_fp8_4096[{target}]: PASS  rel_err={err:.2e}")


@pytest.mark.parametrize("target", TARGETS)
def test_blockscaled_gemm_fp4_e2m1(target):
    err = _test_fp4(gemm, "l1", target=target)
    print(f"test_blockscaled_gemm_fp4_e2m1[{target}]: PASS  rel_err={err:.2e}")


@pytest.mark.parametrize("target", TARGETS)
def test_blockscaled_gemm_l0_fp4_e2m1(target):
    err = _test_fp4(gemm_l0, "l0", target=target)
    print(f"test_blockscaled_gemm_l0_fp4_e2m1[{target}]: PASS  rel_err={err:.2e}")


@pytest.mark.parametrize("target", TARGETS)
def test_blockscaled_gemm_fp4_e2m1_byte_view(target):
    err = _test_fp4_byte_view(target=target)
    print(f"test_blockscaled_gemm_fp4_e2m1_byte_view[{target}]: PASS  rel_err={err:.2e}")


@pytest.mark.parametrize("target", TARGETS)
def test_blockscaled_gemm_fp4_e2m1_8192(target):
    err = _test_fp4(
        gemm,
        "l1",
        M=8192,
        K=8192,
        N=8192,
        seed=123,
        target=target,
    )
    print(f"test_blockscaled_gemm_fp4_e2m1_8192[{target}]: PASS  rel_err={err:.2e}")


@pytest.mark.parametrize("target", TARGETS)
def test_blockscaled_gemm_l0_fp4_e2m1_8192(target):
    err = _test_fp4(
        gemm_l0,
        "l0",
        M=8192,
        K=8192,
        N=8192,
        seed=456,
        target=target,
    )
    print(f"test_blockscaled_gemm_l0_fp4_e2m1_8192[{target}]: PASS  rel_err={err:.2e}")


@pytest.mark.parametrize("target", TARGETS)
def test_blockscaled_gemm_l0_sf_transpose_small_matrices(target):
    for M, K, N in [(15, 128, 15), (32, 128, 32), (100, 128, 100)]:
        err = _test_l0_matrix_sf_transpose(M, K, N, target=target)
        print(f"test_blockscaled_gemm_l0_sf_transpose_small_matrices[{target}] {M}x{K}x{N}: PASS  rel_err={err:.2e}")


if __name__ == "__main__":
    test_blockscaled_gemm_fp8_8192("ascend")
    test_blockscaled_gemm_fp8_4096("ascend")
    test_blockscaled_gemm_l0_fp8_8192("ascend")
    test_blockscaled_gemm_l0_fp8_4096("ascend")
    test_blockscaled_gemm_l0_sf_transpose_small_matrices("ascend")
