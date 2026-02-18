import os
import random
import pytest

os.environ["PYTHONHASHSEED"] = "0"

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
    # Unimplemented sparse ops: tl.tl_gemm_sp / tir.ptx_mma_sp
    "sparse_tensorcore/test_example_sparse_tensorcore.py::test_tilelang_example_sparse_tensorcore",
    "gemm_sp/test_example_gemm_sp.py::test_example_gemm_sp",
    "gemm_sp/test_example_gemm_sp.py::test_example_custom_compress",
    # Flaky — passes when run in isolation, fails under parallel execution
    "minference/test_vs_sparse_attn.py::test_vs_sparse_attn",
}

# Tests that should be skipped entirely under CuTeDSL (arch / feature mismatch).
CUTEDSL_KNOWN_SKIPS = {
    # TMA convolution not fully supported in CuTeDSL
    "convolution/test_example_convolution.py",
    # Stream-K not fully supported in CuTeDSL
    "gemm_streamk/test_example_tilelang_gemm_streamk.py",
    # flash_decoding FA3 variant requires special handling
    "flash_decoding/test_example_flash_decoding.py::test_flash_decoding_fa3",
}


def _match_any(nodeid, patterns):
    """Return True if *nodeid* contains any of the *patterns*."""
    return any(p in nodeid for p in patterns)


def pytest_collection_modifyitems(config, items):
    """When TILELANG_TARGET=cutedsl, annotate known-bad tests automatically."""
    if os.environ.get("TILELANG_TARGET") != "cutedsl":
        return

    for item in items:
        nid = item.nodeid
        if _match_any(nid, CUTEDSL_KNOWN_FAILURES):
            item.add_marker(
                pytest.mark.xfail(
                    reason="CuTeDSL: known limitation (unimplemented op or flaky)",
                    strict=False,
                )
            )
        elif _match_any(nid, CUTEDSL_KNOWN_SKIPS):
            item.add_marker(
                pytest.mark.skip(
                    reason="CuTeDSL: not supported",
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
