# SM120-family block-scaled warp MMA through the CuTeDSL backend.
#
# Runs the kernels of the CUDA block-scaled suites with target="cutedsl":
# NVFP4 (kind::mxf4nvf4 scale_vec::4X, UE4M3), MXFP4 (kind::mxf4nvf4
# scale_vec::2X, UE8M0), MXFP8 (kind::mxf8f6f4 scale_vec::1X, UE8M0) and FP8
# x unpacked E2M1 through ldmatrix .b4x16_p64. The source checks need a
# CuTeDSL-enabled build but no GPU; the runtime checks expect the exact
# products of the CUDA suites on SM120/SM121.

import importlib.util
from pathlib import Path

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.cuda.target import normalize_cutedsl_target


def _is_cutedsl_available():
    try:
        from tilelang.jit.adapter.cutedsl.checks import check_cutedsl_available

        check_cutedsl_available()
        return True
    except (ImportError, AssertionError):
        return False


pytestmark = pytest.mark.skipif(not _is_cutedsl_available(), reason="CuTeDSL not installed")


def _load_suite(name):
    """Reuse the kernels and exact references of a CUDA block-scaled suite."""
    spec = importlib.util.spec_from_file_location(f"_{name}", Path(__file__).with_name(f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mx8 = _load_suite("test_tilelang_language_mxf8f6f4_mma_block_scale")
_mx4 = _load_suite("test_tilelang_language_mxfp4_mma_block_scale")
_nv4 = _load_suite("test_tilelang_language_nvf4_mma_block_scale")


def _make_nvf4_compact_kernel(K):
    return _nv4._make_nvf4_matmul_codegen_kernel(
        128,
        128,
        K,
        num_stages=0,
        block_col_warps=4,
        warp_row_tiles=64,
        sf_layout="blockscaled_chunk_kmajor",
        policy=T.GemmWarpPolicy.FullRow,
    )


_NVF4 = '"m16n8k64", "row", "col", "mxf4nvf4", 4, "e2m1", "e2m1", "ue4m3"'
_MXF4 = '"m16n8k64", "row", "col", "mxf4nvf4", 2, "e2m1", "e2m1", "ue8m0"'


def _mxf8(a_type, b_type):
    return f'"m16n8k32", "row", "col", "mxf8f6f4", 1, "{a_type}", "{b_type}", "ue8m0"'


# name: (kernel builder, MMA configuration, packed FP4 operands, unpacked FP4 B)
_SOURCE_CASES = {
    "nvfp4": (lambda: _nv4._make_nvf4_matmul_codegen_kernel(128, 128, 256), _NVF4, True, False),
    "nvfp4_kmajor": (lambda: _make_nvf4_compact_kernel(256), _NVF4, True, False),
    "mxfp4": (lambda: _mx4._make_mxfp4_matmul_kernel(128, 128, 256), _MXF4, True, False),
    "mxfp8_e4m3": (lambda: _mx8._make_mxfp8_matmul_kernel(128, 128, 256), _mxf8("e4m3", "e4m3"), False, False),
    "mxfp8_e5m2": (
        lambda: _mx8._make_mxfp8_matmul_kernel(128, 128, 256, in_dtype=T.float8_e5m2),
        _mxf8("e5m2", "e5m2"),
        False,
        False,
    ),
    "fp8_x_fp4": (lambda: _mx8._make_mxfp8_matmul_kernel(128, 128, 256, b_fp4=True), _mxf8("e4m3", "e2m1"), False, True),
}


def _cutedsl_kernel_source(prog, arch):
    if not tvm.runtime.enabled("cuda"):
        pytest.skip("TileLang CuTeDSL codegen requires TVM built with CUDA support.")
    if tvm.ffi.get_global_func("target.build.tilelang_cutedsl_without_compile", allow_missing=True) is None:
        pytest.skip("TileLang CuTeDSL backend is not enabled in this build.")
    target = normalize_cutedsl_target({"kind": "cutedsl", "arch": arch})
    with target:
        return lower(prog.with_attr("global_symbol", "main"), target=target).kernel_source


@pytest.mark.parametrize("arch", ["sm_120a", "sm_121a"])
@pytest.mark.parametrize("case", list(_SOURCE_CASES))
def test_mma_block_scale_cutedsl_codegen(case, arch):
    make, config, packed_fp4, unpacked_fp4_b = _SOURCE_CASES[case]
    src = _cutedsl_kernel_source(make(), arch)
    assert f'tl.ptx_mma_block_scale("float32", {config}, ' in src

    ldmatrix = [line for line in src.splitlines() if "tl.ptx_ldmatrix" in line]
    assert ldmatrix
    if unpacked_fp4_b:
        assert any("tl.ptx_ldmatrix_b4x16_p64_x4(" in line for line in ldmatrix)
    else:
        assert "b4x16_p64" not in src
    # FP4 shared rows reach ldmatrix (and cp.async) as byte addresses.
    assert not any("Float4E2M1FN" in line for line in ldmatrix)
    if packed_fp4 or unpacked_fp4_b:
        assert any("dtype=cutlass.Uint8)" in line for line in ldmatrix)
    cp_async = [line for line in src.splitlines() if "tl.cp_async_gs" in line]
    assert not any("Float4E2M1FN" in line for line in cp_async)


@pytest.mark.parametrize(
    "config",
    [
        ("float32", "m16n8k64", "row", "col", "mxf4nvf4", 4, "e2m1", "e2m1", "ue8m0"),
        ("float32", "m16n8k64", "row", "col", "mxf4nvf4", 2, "e4m3", "e2m1", "ue8m0"),
        ("float32", "m16n8k32", "row", "col", "mxf8f6f4", 2, "e4m3", "e4m3", "ue8m0"),
        ("float32", "m16n8k32", "row", "col", "mxf8f6f4", 1, "e4m3", "e4m3", "ue4m3"),
        ("float32", "m16n8k64", "row", "col", "mxf8f6f4", 1, "e4m3", "e4m3", "ue8m0"),
        ("float32", "m16n8k32", "col", "row", "mxf8f6f4", 1, "e4m3", "e4m3", "ue8m0"),
        ("float16", "m16n8k32", "row", "col", "mxf8f6f4", 1, "e4m3", "e4m3", "ue8m0"),
    ],
)
def test_ptx_mma_block_scale_rejects_unsupported_configs(config):
    from tilelang.contrib.cutedsl.mma_block_scale import ptx_mma_block_scale

    with pytest.raises(ValueError, match="Unsupported block-scaled MMA configuration"):
        ptx_mma_block_scale(*config, *([None] * 8))


def _run_cutedsl(prog, *inputs, M, N):
    import torch

    kernel = tilelang.compile(prog, target="cutedsl")
    out = torch.empty((M, N), device="cuda", dtype=torch.float32)
    kernel(*inputs, out)
    return out


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
@pytest.mark.parametrize("fp8_dtype", ["float8_e4m3fn", "float8_e5m2"])
@pytest.mark.parametrize("M, N, K, block_K", [(64, 64, 128, 128), (128, 192, 512, 128), (128, 128, 512, 256)])
def test_mxfp8_cutedsl_correctness(M, N, K, block_K, fp8_dtype):
    import torch

    a, sfa = _mx8._random_mxfp8(M, K, seed=1)
    b, sfb = _mx8._random_mxfp8(N, K, seed=2)
    # The small integers of the E4M3 data are exact in E5M2 as well.
    a, b = (x.float().to(getattr(torch, fp8_dtype)) for x in (a, b))
    prog = _mx8._make_mxfp8_matmul_kernel(M, N, K, block_K, in_dtype=getattr(T, fp8_dtype))
    out = _run_cutedsl(prog, a, b, sfa.view(torch.uint32), sfb.view(torch.uint32), M=M, N=N)
    torch.testing.assert_close(out, _mx8._reference(a, sfa, b, sfb), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
@pytest.mark.parametrize("M, N, K", [(64, 64, 128), (128, 192, 512)])
def test_fp8_x_fp4_cutedsl_correctness(M, N, K):
    import torch

    a, sfa = _mx8._random_mxfp8(M, K, seed=3)
    b_packed, sfb, b_values = _mx8._random_mxfp4(N, K, seed=4)
    prog = _mx8._make_mxfp8_matmul_kernel(M, N, K, b_fp4=True)
    out = _run_cutedsl(prog, a, b_packed, sfa.view(torch.uint32), sfb.view(torch.uint32), M=M, N=N)
    torch.testing.assert_close(out, _mx8._reference(a, sfa, b_values, sfb), rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
@pytest.mark.parametrize("M, N, K", [(64, 64, 128), (128, 192, 512)])
def test_mxfp4_cutedsl_correctness(M, N, K):
    import torch

    a, sfa, a_values = _mx4._random_mxfp4(M, K, seed=5)
    b, sfb, b_values = _mx4._random_mxfp4(N, K, seed=6)
    prog = _mx4._make_mxfp4_matmul_kernel(M, N, K)
    out = _run_cutedsl(prog, a, b, sfa.view(torch.uint32), sfb.view(torch.uint32), M=M, N=N)
    torch.testing.assert_close(out, a_values @ b_values.T, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(12, 0)
@tilelang.testing.requires_cuda_compute_version_lt(13, 0)
@pytest.mark.parametrize("K, compact", [(128, False), (256, False), (128, True), (256, True)])
def test_nvfp4_cutedsl_correctness(K, compact):
    import torch

    torch.manual_seed(0)
    A, B = _nv4._make_packed_fp4_inputs(128, 128, K, "random")
    SFA, sfa_bytes = _nv4._make_varying_power_of_two_scale_words(128, K)
    SFB, sfb_bytes = _nv4._make_varying_power_of_two_scale_words(128, K)
    ref = _nv4._reference_blockscaled_gemm(A, B, sfa_bytes, sfb_bytes, 128, 128, K)
    if compact:
        # The compact package selects scales by lane, through the thread selectors.
        SFA = SFA.reshape(4, 32, K // 64).permute(2, 1, 0).contiguous().reshape(128, K // 64)
        SFB = SFB.reshape(4, 32, K // 64).permute(2, 1, 0).contiguous().reshape(128, K // 64)
        prog = _make_nvf4_compact_kernel(K)
    else:
        prog = _nv4._make_nvf4_matmul_codegen_kernel(128, 128, K)
    out = _run_cutedsl(prog, A, B, SFA, SFB, M=128, N=128)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
