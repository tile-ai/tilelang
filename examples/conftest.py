import os
import random
import pytest

from tilelang.tileir.errors import TileIRLoweringNotImplementedError

os.environ["PYTHONHASHSEED"] = "0"


def _configure_torch_extensions_dir():
    cache_dir = os.environ.get("TILELANG_CACHE_DIR", os.path.expanduser("~/.tilelang/cache"))
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    path = os.path.join(cache_dir, "torch_extension", f"{worker}-{os.getpid()}")
    os.makedirs(path, exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = path


_configure_torch_extensions_dir()

random.seed(0)

try:
    import torch
except ImportError:
    pass
else:
    torch.manual_seed(0)

try:
    import numpy as np
except ImportError:
    pass
else:
    np.random.seed(0)


# ---------------------------------------------------------------------------
# CuTeDSL backend: auto-mark known failures / unsupported tests
# ---------------------------------------------------------------------------

# Known failures when running with TILELANG_TARGET=cutedsl.
# These are marked as xfail(strict=False) so unexpected passes are reported.
CUTEDSL_KNOWN_FAILURES = {
    # Flaky — passes when run in isolation, fails under parallel execution
    "minference/test_vs_sparse_attn.py::test_vs_sparse_attn",
    # CuTeDSL does not yet lower DeepSeek V4 FP4 act quant conversions.
    "deepseek_v4/test_tilelang_example_deepseek_v4.py::test_example_act_quant",
}

# Known limitations when running the examples with TILELANG_TARGET=tileir.
TILEIR_KNOWN_FAILURES = {
    # Blackwell cluster-specialized tcgen path uses direct PTX/cluster intrinsics
    # (tcgen05 warp copies, cluster barriers, and fence_proxy_async).
    "deepseek_v4/test_tilelang_example_deepseek_v4.py::test_example_fp8_fp4_gemm_1d1d",
    # Per-thread SIMT dequant kernels (thread-allreduce / threadIdx gathers) — not in the collective tile model.
    "dequantize_gemm/test_example_dequantize_gemm.py::test_example_dequant_gemv_fp16xint4",
    "dequantize_gemm/test_example_dequantize_gemm.py::test_example_dequant_gemm_bf16_mxfp4_hopper",
    # Explicit PTX intrinsic path, outside structured lowering.
    "gemm/test_example_gemm.py::test_example_gemm_intrinsics",
    # 2:4 sparse MMA (gemm_sp): no sparse MMA op in the cuda_tile dialect.
    "gemm_sp/test_example_gemm_sp.py::test_example_gemm_sp",
    # Scalar shared-memory indexed stores are rejected.
    "gemv/test_example_gemv.py::test_example_gemv",
    # Non-power-of-two tile dims (192-head, 576-wide dQ) — need the pad/split engine.
    "flash_attention/test_example_flash_attention.py::test_example_gqa_bwd",
    "flash_attention/test_example_flash_attention.py::test_example_gqa_bwd_tma_reduce_varlen",
    "deepseek_v32/test_tilelang_example_deepseek_v32.py::test_example_sparse_mla_bwd",
    # Non-power-of-two fragment (24-wide) slicing + SIMT shared prefix sums.
    "deepseek_mhc/test_example_mhc.py::test_mhc_pre",
    # Warp-specialized ptx_cp_async / set_max_nreg — no structured counterpart.
    "deepseek_v32/test_tilelang_example_deepseek_v32.py::test_example_sparse_mla_fwd_pipelined",
    # Non-power-of-two 257-bin histogram + cross-lane prefix scan (scatter/return-atomics do lower).
    "deepseek_v32/test_tilelang_example_deepseek_v32.py::test_example_topk_selector",
}


def _match_any(nodeid, patterns):
    """Return True if *nodeid* contains any of the *patterns*."""
    return any(p in nodeid for p in patterns)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Annotate backend-specific known-bad example tests automatically."""
    target = os.environ.get("TILELANG_TARGET", "").lower()

    if target == "cutedsl":
        for item in items:
            nid = item.nodeid
            if _match_any(nid, CUTEDSL_KNOWN_FAILURES):
                item.add_marker(
                    pytest.mark.xfail(
                        reason="CuTeDSL: known limitation (unimplemented op or flaky)",
                        strict=False,
                    )
                )
        return

    if target.startswith("tileir"):
        for item in items:
            nid = item.nodeid
            if _match_any(nid, TILEIR_KNOWN_FAILURES):
                item.add_marker(
                    pytest.mark.xfail(
                        reason="TileIR: known structured lowering limitation",
                        raises=TileIRLoweringNotImplementedError,
                        strict=True,
                    )
                )


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Ensure that at least one test is collected. Error out if all tests are skipped."""
    known_types = {
        "failed",
        "passed",
        "skipped",
        "deselected",
        "xfailed",
        "xpassed",
        "warnings",
        "error",
    }
    if sum(len(terminalreporter.stats.get(k, [])) for k in known_types.difference({"skipped", "deselected"})) == 0:
        terminalreporter.write_sep(
            "!",
            (f"Error: No tests were collected. {dict(sorted((k, len(v)) for k, v in terminalreporter.stats.items()))}"),
        )
        pytest.exit("No tests were collected.", returncode=5)
