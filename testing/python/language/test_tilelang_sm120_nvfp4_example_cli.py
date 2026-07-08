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
    ]
    for flag in forbidden_flags:
        assert flag not in source
