"""Device numeric tests for the float Min/Max special-value contract.

``Min(a, b)`` / ``Max(a, b)`` legalize to ``Select(a < b, a, b)`` /
``Select(a > b, a, b)``, whose behaviour on IEEE special values is *not*
``fmin``/``fmax``:

* a NaN operand in the *left* position makes the comparison false, so the right
  operand wins and the NaN disappears;
* a NaN operand in the *right* position is returned as-is when the left one is
  not smaller/greater (any comparison with NaN is false), so the NaN survives;
* ``-0`` vs ``+0`` compare equal, so the *right* operand wins and the sign
  follows operand order.

Both operations run in one kernel (row 0 is ``Min``, row 1 is ``Max``), so the
contract is pinned by a single device compile.

These tests pin that behaviour so the documented contract cannot drift.
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

E, LANES = 256, 64


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
    """One kernel, two units: row 0 is ``Min``, row 1 is ``Max``."""

    @T.prim_func
    def main(A: T.Tensor((E,), "float32"), B: T.Tensor((E,), "float32"), Y: T.Tensor((2, E), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((2, E), "float32")
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.Parallel(E):
                    y_ub[0, i] = T.min(a_ub[i], b_ub[i])
                for i in T.Parallel(E):
                    y_ub[1, i] = T.max(a_ub[i], b_ub[i])
            T.copy(y_ub, Y)

    return main


def _ref(a, b, op):
    """Select(a<b,a,b) / Select(a>b,a,b) — numpy's where has the same
    NaN semantics: the comparison is false, so the b branch is taken."""
    return np.where(a < b, a, b) if op == "min" else np.where(a > b, a, b)


_KERNEL = [None]


def _kernel():
    """Compile the pair kernel once: a device compile dominates the cost."""
    if _KERNEL[0] is None:
        _KERNEL[0] = tilelang.compile(_build(), target="pto", out_idx=-1)
    return _KERNEL[0]


def _run(a, b):
    kernel = _kernel()
    got = kernel(torch.from_numpy(a).to(_dev()), torch.from_numpy(b).to(_dev())).cpu().numpy().astype(np.float32)
    for op, row in (("min", 0), ("max", 1)):
        want = _ref(a, b, op).astype(np.float32)
        bad_mask = ~np.isclose(got[row], want, rtol=0.0, atol=0.0, equal_nan=True)
        bad = int(bad_mask.sum())
        assert bad == 0, f"{op}: {bad}/{E} mismatched; first: {np.flatnonzero(bad_mask)[:5]}"
    return got


@pytest.mark.pto
@requires_npu
def test_nan_takes_the_right_operand_position():
    """NaN on the left is dropped, NaN on the right survives (Min and Max)."""
    a = np.array([np.nan, 1.0, np.nan, 2.0] * (E // 4), dtype=np.float32)
    b = np.array([1.0, np.nan, np.nan, 2.0] * (E // 4), dtype=np.float32)
    got = _run(a, b)
    for op, row in (("min", 0), ("max", 1)):
        # left NaN -> b (finite), right NaN -> NaN
        assert not np.isnan(got[row][0]), f"{op}: NaN on the left must not survive"
        assert np.isnan(got[row][1]), f"{op}: NaN on the right must survive"
        assert np.isnan(got[row][2]), f"{op}: NaN on both sides must survive"
        assert got[row][3] == np.float32(2.0), f"{op}: finite pair must be exact"


@pytest.mark.pto
@requires_npu
def test_negative_zero_follows_operand_order():
    """+0 and -0 compare equal, so the right operand wins (Min and Max)."""
    pos = np.float32(0.0)
    neg = np.float32(-0.0)
    a = np.array([pos, neg] * (E // 2), dtype=np.float32)
    b = np.array([neg, pos] * (E // 2), dtype=np.float32)
    got = _run(a, b)
    for op, row in (("min", 0), ("max", 1)):
        # (a, b) = (+0, -0) -> picks -0 ; (a, b) = (-0, +0) -> picks +0
        assert np.signbit(got[row][0]), f"{op}: (+0, -0) must return -0"
        assert not np.signbit(got[row][1]), f"{op}: (-0, +0) must return +0"
