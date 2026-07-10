import importlib.util
import sys
from pathlib import Path


def _load_sm120_example(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    example = repo_root / "examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py"
    monkeypatch.setattr(sys, "argv", [str(example)])
    spec = importlib.util.spec_from_file_location("sm120_nvfp4_blockscaled_gemm_example", example)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_sm120_nvfp4_example_cli_keeps_internal_strategy_fixed(monkeypatch):
    module = _load_sm120_example(monkeypatch)
    args = module.parse_args()

    assert not hasattr(args, "micro_pipeline")
    assert not hasattr(args, "sf_layout")
    assert not hasattr(args, "scale_storage_layout")
    assert not hasattr(args, "manual_ws2")

    assert module._SM120_SCALE_LAYOUT == "blockscaled_chunk_kmajor"


def test_sm120_nvfp4_example_source_has_no_legacy_strategy_flags():
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "examples/gemm_sm120/sm120_nvfp4_blockscaled_gemm.py").read_text()

    forbidden_flags = [
        "--micro-pipeline",
        "--manual-ws2",
        "--manual-ws2-split",
        "--manual-ws2-sf-load",
        "--manual-ws2-sf-layout",
        "--manual-ws2-ab-shared-storage",
        "--manual-ws2-ab-copy-view",
        "--scale-storage-layout",
        "--sf-layout",
        "--path",
        "--run-cutlass",
        "--cutlass-build-dir",
        "--cutlass-binary",
        "--rebuild-cutlass",
        "--cmake",
        "--nvcc",
    ]
    for flag in forbidden_flags:
        assert flag not in source

    forbidden_harness_terms = [
        "build_cutlass",
        "subprocess",
        "CMAKE_CUDA_COMPILER",
    ]
    for term in forbidden_harness_terms:
        assert term not in source


def test_sm120_nvfp4_example_kernel_handles_mn_tail_tiles(monkeypatch):
    import pytest

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (12, 0):
        pytest.skip("requires an SM120 GPU")

    from tilelang.quantize import swizzle_blockscaled_chunk_kmajor_scale_words

    module = _load_sm120_example(monkeypatch)
    for M, N, K in [(257, 384, 512), (130, 128, 256), (128, 136, 256)]:
        kernel = module.sm120_nvfp4_blockscaled_gemm(M, N, K)

        A = module._make_packed_fp4(M, K, seed=3)
        B = module._make_packed_fp4(N, K, seed=4)
        SFA_semantic = module._make_binary_scale_words(M, K, seed=5)
        SFB_semantic = module._make_binary_scale_words(N, K, seed=6)
        SFA = swizzle_blockscaled_chunk_kmajor_scale_words(SFA_semantic)
        SFB = swizzle_blockscaled_chunk_kmajor_scale_words(SFB_semantic)
        assert SFA.shape[0] % 128 == 0
        assert SFB.shape[0] % 128 == 0

        C = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
        kernel(A, B, SFA, SFB, C)
        torch.cuda.synchronize()
        module._verify(A, B, SFA_semantic, SFB_semantic, C, torch.bfloat16)


def test_sm120_nvfp4_example_kernel_rejects_unsupported_tails(monkeypatch):
    import pytest

    module = _load_sm120_example(monkeypatch)
    # simultaneous M and N tails hit a known copy-lowering boundary bug
    with pytest.raises(ValueError, match="simultaneous M and N tail"):
        module.sm120_nvfp4_blockscaled_gemm(257, 136, 512)
    # bf16 output rows must stay 16-byte aligned
    with pytest.raises(AssertionError, match="multiple of 8"):
        module.sm120_nvfp4_blockscaled_gemm(128, 130, 256)
