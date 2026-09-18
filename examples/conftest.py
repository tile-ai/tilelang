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
    "deepseek_v4/test_tilelang_example_deepseek_v4.py::test_example_act_quant_fp4",
}

# Backend limitations, not a list of CUDA 13.3 compiler bugs. Keep each reason
# visible in pytest's xfail report. Architecture skips still take precedence;
# a skipped test does not confirm whether its lowering limitation remains.
TILEIR_KNOWN_FAILURES = {
    "deepseek_v4/test_tilelang_example_deepseek_v4.py::test_example_fp8_fp4_gemm_1d1d": (
        "tir.assume is not lowered; the kernel also uses direct tcgen05 copies and cluster intrinsics"
    ),
    "dequantize_gemm/test_example_dequantize_gemm.py::test_example_dequant_gemv_fp16xint4": (
        "thread-local access_ptr lowering fails for the external CUDA dequantization helper"
    ),
    "dequantize_gemm/test_example_dequantize_gemm.py::test_example_dequant_gemm_bf16_mxfp4_hopper": (
        "per-thread SIMT dequantization requires thread-indexed gathers and reductions"
    ),
    "gemm/test_example_gemm.py::test_example_gemm_intrinsics": "explicit tl.ptx_ldmatrix / tir.ptx_mma have no structured lowering",
    "gemm_sp/test_example_gemm_sp.py::test_example_gemm_sp": "2:4 sparse MMA has no CUDA Tile IR counterpart",
    "gemv/test_example_gemv.py::test_example_gemv": "per-thread SIMT scatter into shared memory is not supported",
    "flash_attention/test_example_flash_attention.py::test_example_gqa_bwd": "192-wide TileViews require non-power-of-two lowering",
    "flash_attention/test_example_flash_attention.py::test_example_gqa_bwd_tma_reduce_varlen": (
        "non-power-of-two attention tiles require padding or splitting"
    ),
    "deepseek_v32/test_tilelang_example_deepseek_v32.py::test_example_sparse_mla_bwd": (
        "576-wide gradient tiles require non-power-of-two lowering"
    ),
    "deepseek_v32/test_tilelang_example_deepseek_v32.py::test_example_sparse_mla_fwd_pipelined": (
        "warp-specialized ptx_cp_async and manual barriers have no structured lowering"
    ),
    "deepseek_v32/test_tilelang_example_deepseek_v32.py::test_example_topk_selector": (
        "257-bin histogram tiles and cross-lane prefix scans are not supported"
    ),
    "minference/test_vs_sparse_attn.py::test_vs_sparse_attn": (
        "indirect K/V gathers through staged shared-memory column indices are not lowered"
    ),
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
            # Match an exact test, including all of its parameterizations, but
            # never another test whose name merely extends the same prefix.
            nid = item.nodeid.split("[", 1)[0]
            for pattern, reason in TILEIR_KNOWN_FAILURES.items():
                if nid == pattern or nid.endswith("/" + pattern):
                    item.add_marker(
                        pytest.mark.xfail(
                            reason=f"TileIR: {reason}",
                            raises=TileIRLoweringNotImplementedError,
                            strict=True,
                        )
                    )
                    break


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
