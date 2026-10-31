"""Device numeric test for a conditional destination with a *runtime* offset.

A conditional write folds into a value choice: the side that does not write reads
the destination back, so the destination must stay per-lane. That requirement is
the continuity proof (``start + lane`` over the whole chunk), not a constant
offset: ``i + 8 * K`` with a runtime ``K`` keeps the lane coefficient at 1 and the
read-back per-lane, so it is admitted.

This test pins the numeric result of that shape on the device: only lanes whose
condition holds are written, at the shifted window, and every other lane keeps
its initial value. It also confirms the shifted window's 32-byte alignment holds
for a runtime offset (``8 * K`` floats = ``32 * K`` bytes).
"""

import contextlib
import os
import sys

import numpy as np
import pytest
import torch

with contextlib.suppress(ImportError):
    # The CPU wheel has no `torch.npu`; the device image registers it here.
    import torch_npu  # noqa: F401

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ["TILELANG_DISABLE_DATA_RACE_CHECK"] = "1"

import tilelang
import tilelang.ascend.language as T

E, LANES, WIDTH = 256, 64, 512
SENTINEL = np.float32(-5.0)


def _npu_available():
    if not hasattr(torch, "npu"):
        return False
    try:
        torch.randn(1, device=torch.device("npu"))
        torch.npu.synchronize()
        return True
    except Exception:  # noqa: BLE001
        return False


_NPU = _npu_available()
requires_npu = pytest.mark.skipif(not _NPU, reason="NPU runtime is unavailable")


def _dev():
    return torch.device("npu")


def _build():
    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        INIT: T.Tensor((WIDTH,), "float32"),
        Y: T.Tensor((WIDTH,), "float32"),
        K: T.int32,
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((WIDTH,), "float32")
            T.copy(A, a_ub)
            T.copy(INIT, y_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.Parallel(E):
                    if a_ub[i] > T.float32(0.0):
                        y_ub[i + 8 * K] = a_ub[i]
            T.copy(y_ub, Y)

    return main


_KERNEL = [None]


def _kernel():
    if _KERNEL[0] is None:
        # No `out_idx`: the kernel mixes a scalar parameter with buffers, and the
        # output index resolves through the buffer map. Y is written in place.
        _KERNEL[0] = tilelang.compile(_build(), target="pto")
    return _KERNEL[0]


def _golden(a, k):
    y = np.full(WIDTH, SENTINEL, dtype=np.float32)
    lanes = np.arange(E)
    taken = a > 0.0
    y[lanes[taken] + 8 * k] = a[taken]
    return y


def _run(a, k):
    init = np.full(WIDTH, SENTINEL, dtype=np.float32)
    out = np.zeros(WIDTH, dtype=np.float32)
    kernel = _kernel()
    out_dev = torch.from_numpy(out).to(_dev())
    kernel(
        torch.from_numpy(a).to(_dev()),
        torch.from_numpy(init).to(_dev()),
        out_dev,
        int(k),
    )
    got = out_dev.cpu().numpy().astype(np.float32)
    want = _golden(a, k)
    bad_mask = ~np.isclose(got, want, rtol=0.0, atol=0.0, equal_nan=True)
    bad = int(bad_mask.sum())
    assert bad == 0, f"k={k}: {bad}/{WIDTH} mismatched; first: {np.flatnonzero(bad_mask)[:5]}"
    return got


def _mixed_input():
    """Alternating positive and negative lanes with distinct magnitudes."""
    a = np.empty(E, dtype=np.float32)
    a[0::2] = np.arange(1, E // 2 + 1, dtype=np.float32)
    a[1::2] = -np.arange(1, E // 2 + 1, dtype=np.float32)
    return a


@pytest.mark.pto
@requires_npu
@pytest.mark.parametrize("k", [0, 1, 4])
def test_runtime_offset_window_is_written_per_lane(k):
    """Only the lanes whose condition holds write, at the shifted window."""
    _run(_mixed_input(), k)


@pytest.mark.pto
@requires_npu
def test_runtime_offset_keeps_the_false_lanes():
    """A lane whose condition fails keeps its slot; a passing lane writes its own."""
    a = _mixed_input()
    got = _run(a, 4)
    # lane 0 is positive (1.0): it writes index 0 + 32
    assert got[32] == np.float32(a[0]), "a passing lane must write its own slot"
    # lane 1 is negative: its window slot stays at the initial value
    assert got[33] == SENTINEL, "a failing lane must leave its slot untouched"
