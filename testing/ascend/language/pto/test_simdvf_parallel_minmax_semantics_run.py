"""Device numeric tests for the float Min/Max special-value contract.

Both configurations of ``tl.pto.relaxed_minmax`` are pinned, because they
disagree on IEEE special values:

* ``False`` — Legalize rewrites ``Min(a, b)`` / ``Max(a, b)`` into
  ``Select(a < b, a, b)`` / ``Select(a > b, a, b)``, whose behaviour is *not*
  ``fmin``/``fmax``:

  - a NaN operand in the *left* position makes the comparison false, so the
    right operand wins and the NaN disappears;
  - a NaN operand in the *right* position is returned as-is when the left one
    is not smaller/greater (any comparison with NaN is false), so the NaN
    survives;
  - ``-0`` vs ``+0`` compare equal, so the *right* operand wins and the sign
    follows operand order.

* ``True`` (default) — the direct VMI instruction follows the hardware
  Min/Max semantics: a NaN operand propagates from either side, and the
  signed-zero difference is ignored (``min`` returns ``-0``, ``max`` returns
  ``+0``).

Both operations run in one kernel (row 0 is ``Min``, row 1 is ``Max``), so each
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


_KERNELS = {}


def _kernel(relaxed):
    """Compile one kernel per configuration: a device compile dominates cost."""
    if relaxed not in _KERNELS:
        _KERNELS[relaxed] = tilelang.compile(
            _build(),
            target="pto",
            out_idx=-1,
            pass_configs={
                tilelang.PassConfigKey.TL_PTO_RELAXED_MINMAX: relaxed,
                tilelang.PassConfigKey.TL_DISABLE_DATA_RACE_CHECK: True,
            },
        )
    return _KERNELS[relaxed]


def _run(a, b, relaxed):
    kernel = _kernel(relaxed)
    got = kernel(torch.from_numpy(a).to(_dev()), torch.from_numpy(b).to(_dev())).cpu().numpy().astype(np.float32)
    if not relaxed:
        # The Select form has a numpy reference; the direct instruction does not
        # (NaN propagation and signed-zero handling differ), so only the
        # special-value assertions below pin it.
        for op, row in (("min", 0), ("max", 1)):
            want = _ref(a, b, op).astype(np.float32)
            bad_mask = ~np.isclose(got[row], want, rtol=0.0, atol=0.0, equal_nan=True)
            bad = int(bad_mask.sum())
            assert bad == 0, f"{op}: {bad}/{E} mismatched; first: {np.flatnonzero(bad_mask)[:5]}"
    return got


@pytest.mark.pto
@requires_npu
@pytest.mark.parametrize("relaxed", [False, True])
def test_nan_operand_handling(relaxed):
    """Where the NaN survives depends on the lowering: Select drops a left NaN,
    the direct instruction propagates NaN from either side."""
    a = np.array([np.nan, 1.0, np.nan, 2.0] * (E // 4), dtype=np.float32)
    b = np.array([1.0, np.nan, np.nan, 2.0] * (E // 4), dtype=np.float32)
    got = _run(a, b, relaxed)
    for op, row in (("min", 0), ("max", 1)):
        if relaxed:
            assert np.isnan(got[row][0]), f"{op}: NaN on the left must survive"
        else:
            assert not np.isnan(got[row][0]), f"{op}: NaN on the left must not survive"
        assert np.isnan(got[row][1]), f"{op}: NaN on the right must survive"
        assert np.isnan(got[row][2]), f"{op}: NaN on both sides must survive"
        assert got[row][3] == np.float32(2.0), f"{op}: finite pair must be exact"


@pytest.mark.pto
@requires_npu
@pytest.mark.parametrize("relaxed", [False, True])
def test_zero_pair_sign_handling(relaxed):
    """Select lets the right operand win the ``+0`` vs ``-0`` tie; the direct
    instruction ignores the difference (``min`` picks ``-0``, ``max`` picks
    ``+0``)."""
    pos = np.float32(0.0)
    neg = np.float32(-0.0)
    a = np.array([pos, neg] * (E // 2), dtype=np.float32)
    b = np.array([neg, pos] * (E // 2), dtype=np.float32)
    got = _run(a, b, relaxed)
    # The value must be zero, not merely signed: `min = -1` / `max = +1` would
    # also satisfy the sign checks below.
    for op, row in (("min", 0), ("max", 1)):
        assert np.all(got[row] == np.float32(0.0)), f"{op}: a zero pair must return zero, got {got[row][:4]}"
    if relaxed:
        assert np.signbit(got[0][0]) and np.signbit(got[0][1]), "min must return -0 for a zero pair"
        assert not np.signbit(got[1][0]) and not np.signbit(got[1][1]), "max must return +0 for a zero pair"
    else:
        for op, row in (("min", 0), ("max", 1)):
            # (a, b) = (+0, -0) -> picks -0 ; (a, b) = (-0, +0) -> picks +0
            assert np.signbit(got[row][0]), f"{op}: (+0, -0) must return -0"
            assert not np.signbit(got[row][1]), f"{op}: (-0, +0) must return +0"
