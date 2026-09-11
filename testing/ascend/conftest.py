"""Gate the Ascend suite on a build that includes the Ascend backend.

The Ascend sources are gated behind USE_ASCEND, so a CUDA-only build has no
Ascend target, no tl.ascend_*/tl.simd.* ops and no Ascend pass-config options.
Collecting these tests there produces a wall of unrelated failures instead of one
clear statement that the backend is not in the library.

The gate has to act at collection time rather than through a per-item mark.
Several modules do backend work at import scope -- creating an "ascend" Target,
looking up tl.simd.*/tl.ascend_* ops, importing torch_npu -- which raises while
pytest is still importing them, before any mark could apply. Ignoring the
directory keeps those imports from happening at all.

Per-test marks still apply on top of this: requires_ascend also carries the
run-time (device present) check, which this hook deliberately does not make, so
source-only lowering tests keep running on a host without an NPU.
"""

from __future__ import annotations

import pytest

import tilelang.testing

SKIP_REASON = "Ascend backend is not built into this library (configure with USE_ASCEND=ON)"

ASCEND_SUITE_GATED = not tilelang.testing.ascend_backend_compiled()

if ASCEND_SUITE_GATED:
    collect_ignore_glob = ["*"]


def pytest_collection_modifyitems(config, items):
    if ASCEND_SUITE_GATED:
        # testing/conftest.py turns this into a one-line note instead of its
        # "No tests were collected" error, which is meant for misconfigured runs.
        config._ascend_suite_gated = True
