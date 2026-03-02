import os
import random
import pytest
import contextlib

# Keep deterministic behavior consistent with the original configuration
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
# Enable TVM pytest plugin so CUDA/GPU markers are registered globally here.
# This fixes Unknown mark warnings and allows marker-based filtering.
# ---------------------------------------------------------------------------
pytest_plugins = ["tvm.testing.plugin"]


# ---------------------------------------------------------------------------
# CuTeDSL backend: auto-mark known failures / unsupported tests (original logic)
# ---------------------------------------------------------------------------

# Known failures when running with TILELANG_TARGET=cutedsl.
# These are marked as xfail(strict=False) so unexpected passes are reported.
CUTEDSL_KNOWN_FAILURES = {
    # Unimplemented sparse ops: tl.tl_gemm_sp
    "sparse_tensorcore/test_example_sparse_tensorcore.py::test_tilelang_example_sparse_tensorcore",
    "gemm_sp/test_example_gemm_sp.py::test_example_gemm_sp",
    # Flaky — passes when run in isolation, fails under parallel execution
    "minference/test_vs_sparse_attn.py::test_vs_sparse_attn",
}


def _match_any(nodeid, patterns):
    """Return True if nodeid contains any of the patterns."""
    return any(p in nodeid for p in patterns)


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
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


def pytest_terminal_summary(terminalreporter, exitstatus, config):  # noqa: ARG001
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


# ---------------------------------------------------------------------------
# CUDA synchronization around tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cuda_sync_around_test(request):
    """Synchronize CUDA before and after each test to surface async errors early.

    Behavior:
    - If CUDA is unavailable, does nothing.
    - By default (no env set), runs for all tests when CUDA is available.
    - If TILELANG_SYNC_CUDA_ONLY_MARKED=1, only runs for tests marked with
      either `cuda` or `gpu` (e.g. via tilelang.testing.requires_cuda or
      tvm.testing.requires_cuda).
    """

    try:
        import torch as _torch  # local import to avoid import-time failure
    except Exception:
        yield
        return

    if not _torch.cuda.is_available():
        yield
        return

    restrict_to_marked = os.environ.get("TILELANG_SYNC_CUDA_ONLY_MARKED", "").lower() in {
        "1",
        "true",
        "yes",
    }

    if restrict_to_marked:
        # Only sync if the test is explicitly marked as using CUDA/GPU.
        marked = bool(request.node.get_closest_marker("cuda") or request.node.get_closest_marker("gpu"))
        if not marked:
            yield
            return

    # Pre-test sync: if a prior test left an async error pending in this worker,
    # this will raise here instead of corrupting the next test.
    with contextlib.suppress(Exception):
        _torch.cuda.synchronize()

    try:
        yield
    finally:
        # Post-test sync: flush any device work so errors don't spill into the next test.
        with contextlib.suppress(Exception):
            _torch.cuda.synchronize()
