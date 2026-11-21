"""Device numeric tests for conditional writes in a converting region.

A conditional write (``if cond: B[i] = v``) is lowered into an unpredicated
store whose value is ``Select(cond, v, B[i])``: the missing side reads the
destination back. A vecscope has no ordering point, so that read observes
whatever UB held at that point, including a value written by an earlier unit
of the same region. These tests pin the numeric result of that contract,
including behind a UB polluter, which makes the read value deterministic
instead of incidental.

All shapes run in one kernel so a single device compile pins the whole set:

======================  ====================================================
row                     case
======================  ====================================================
0                       one-sided write (read-back keeps the old value)
1                       nested one-sided (``x > 0 and z > 0``)
2                       ``or`` condition
3                       nested one-sided inside a then/else pair
4, 5                    arm-local write-then-read (two destinations)
6                       bare scalar bool condition
7                       carried buffer, one-sided update inside a serial K
======================  ====================================================

Rows 0-3 and 6 share one ``T.Parallel`` body (several independent
conditionals in a single unit); rows 4-5 and 7 use their own units.
"""

import contextlib
import os
import sys
from functools import lru_cache

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
K = 8
POLLUTE_N = 8192
RTOL, ATOL = 1e-5, 1e-6

ROW_SINGLE_SIDED = 0
ROW_NESTED = 1
ROW_OR = 2
ROW_NESTED_THEN_ELSE = 3
ROW_ARM0 = 4
ROW_ARM1 = 5
ROW_SCALAR_BOOL = 6
ROW_CARRIED = 7
ROWS = 8


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


def _build_polluter(n=POLLUTE_N):
    """Paint the UB window with 777.0 and copy it out."""

    @T.prim_func
    def main(Out: T.Tensor((n,), "float32")):
        with T.Kernel(1):
            buf = T.alloc_shared((n,), "float32")
            with T.SimdVF(lanes=64):
                for i in T.Parallel(n):
                    buf[i] = T.float32(777.0)
            T.copy(buf, Out)

    return main


def _build_branch_shapes():
    """One kernel, all conditional-write shapes (see the module docstring)."""

    @T.prim_func
    def main(
        X: T.Tensor((E,), "float32"),
        Z: T.Tensor((E,), "float32"),
        A: T.Tensor((E,), "float32"),
        B: T.Tensor((E,), "float32"),
        Thresh: T.Tensor((K,), "float32"),
        flag: T.bool,
        Y: T.Tensor((ROWS, E), "float32"),
    ):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            z_ub = T.alloc_shared((E,), "float32")
            a_ub = T.alloc_shared((E,), "float32")
            b_ub = T.alloc_shared((E,), "float32")
            thr_ub = T.alloc_shared((K,), "float32")
            y_ub = T.alloc_shared((ROWS, E), "float32")
            T.copy(X, x_ub)
            T.copy(Z, z_ub)
            T.copy(A, a_ub)
            T.copy(B, b_ub)
            T.copy(Thresh, thr_ub)
            with T.SimdVF(lanes=LANES):
                # Several independent conditionals in one unit.
                for i in T.Parallel(E):
                    y_ub[ROW_SINGLE_SIDED, i] = T.float32(7.0)
                    y_ub[ROW_NESTED, i] = T.float32(7.0)
                    y_ub[ROW_OR, i] = T.float32(7.0)
                    y_ub[ROW_NESTED_THEN_ELSE, i] = T.float32(7.0)
                    if x_ub[i] > T.float32(0.0):
                        y_ub[ROW_SINGLE_SIDED, i] = T.float32(1.0)
                    # The nesting is the shape under test: folding a nested
                    # branch differs from folding one conjunction.
                    if x_ub[i] > T.float32(0.0):  # noqa: SIM102
                        if z_ub[i] > T.float32(0.0):
                            y_ub[ROW_NESTED, i] = T.float32(1.0)
                    if (x_ub[i] > T.float32(0.0)) or (z_ub[i] > T.float32(0.0)):
                        y_ub[ROW_OR, i] = T.float32(1.0)
                    if a_ub[i] > T.float32(0.0):
                        if b_ub[i] > T.float32(0.0):
                            y_ub[ROW_NESTED_THEN_ELSE, i] = T.float32(1.0)
                    else:
                        y_ub[ROW_NESTED_THEN_ELSE, i] = T.float32(2.0)
                    if flag:
                        y_ub[ROW_SCALAR_BOOL, i] = a_ub[i]
                    else:
                        y_ub[ROW_SCALAR_BOOL, i] = b_ub[i]
                # Arm-local write-then-read on two destinations.
                for i in T.Parallel(E):
                    if a_ub[i] > T.float32(0.0):
                        y_ub[ROW_ARM0, i] = a_ub[i]
                        y_ub[ROW_ARM1, i] = y_ub[ROW_ARM0, i] * T.float32(2.0)
                    else:
                        y_ub[ROW_ARM1, i] = a_ub[i]
                        y_ub[ROW_ARM0, i] = y_ub[ROW_ARM1, i] * T.float32(3.0)
                # Carried buffer, one-sided update inside a serial K loop.
                acc = T.alloc_shared((E,), "float32")
                for i in T.Parallel(E):
                    acc[i] = T.float32(0.0)
                for k in T.serial(K):
                    thr = thr_ub[k]
                    for i in T.Parallel(E):
                        if x_ub[i] > thr:
                            acc[i] = acc[i] + T.float32(1.0)
                for i in T.Parallel(E):
                    y_ub[ROW_CARRIED, i] = acc[i]
            T.copy(y_ub, Y)

    return main


# ---------------------------------------------------------------------------
# goldens
# ---------------------------------------------------------------------------


def _gold_single_sided(x):
    return np.where(x > 0, np.float32(1.0), np.float32(7.0)).astype(np.float32)


def _gold_nested(x, z):
    return np.where((x > 0) & (z > 0), np.float32(1.0), np.float32(7.0)).astype(np.float32)


def _gold_nested_then_else(a, b):
    return np.where(a > 0, np.where(b > 0, np.float32(1.0), np.float32(7.0)), np.float32(2.0)).astype(np.float32)


def _gold_or(x, z):
    return np.where((x > 0) | (z > 0), np.float32(1.0), np.float32(7.0)).astype(np.float32)


def _gold_arm_read_after_write(a):
    y0 = np.where(a > 0, a, a * 3.0)
    y1 = np.where(a > 0, a * 2.0, a)
    return np.stack([y0, y1]).astype(np.float32)


def _gold_carried(x, thr):
    acc = np.zeros_like(x)
    for k in range(len(thr)):
        acc = np.where(x > np.float32(thr[k]), acc + np.float32(1.0), acc).astype(np.float32)
    return acc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _compile(prim):
    return tilelang.compile(prim, target="pto", out_idx=-1)


@lru_cache(maxsize=None)
def _kernel(builder):
    """Compile each builder once: a device compile dominates the test cost."""
    return _compile(builder())


def _check_row(got, row, gold, tag):
    bad_mask = ~np.isclose(got[row], gold, rtol=RTOL, atol=ATOL)
    bad = int(bad_mask.sum())
    assert bad == 0, f"{tag}: {bad}/{len(gold)} mismatched; first: {np.flatnonzero(bad_mask)[:5]}"


def _inputs():
    rng_x = np.random.default_rng(0)
    rng_z = np.random.default_rng(2)
    rng_a = np.random.default_rng(4)
    rng_b = np.random.default_rng(5)
    x = rng_x.uniform(-2.0, 2.0, E).astype(np.float32)
    z = rng_z.uniform(-2.0, 2.0, E).astype(np.float32)
    a = rng_a.uniform(-2.0, 2.0, E).astype(np.float32)
    b = rng_b.uniform(-2.0, 2.0, E).astype(np.float32)
    thr = np.linspace(-1.5, 1.5, K).astype(np.float32)
    return x, z, a, b, thr


def _run(kernel, x, z, a, b, thr, flag):
    return (
        kernel(
            torch.from_numpy(x).to(_dev()),
            torch.from_numpy(z).to(_dev()),
            torch.from_numpy(a).to(_dev()),
            torch.from_numpy(b).to(_dev()),
            torch.from_numpy(thr).to(_dev()),
            flag,
        )
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


@pytest.mark.pto
@requires_npu
def test_branch_shapes_numeric():
    """Every conditional shape selects the right lane values (both flags)."""
    x, z, a, b, thr = _inputs()
    kernel = _kernel(_build_branch_shapes)

    for flag in (True, False):
        got = _run(kernel, x, z, a, b, thr, flag)
        tag = f"flag={flag}"
        _check_row(got, ROW_SINGLE_SIDED, _gold_single_sided(x), f"{tag} single-sided")
        _check_row(got, ROW_NESTED, _gold_nested(x, z), f"{tag} nested")
        _check_row(got, ROW_OR, _gold_or(x, z), f"{tag} or-condition")
        _check_row(got, ROW_NESTED_THEN_ELSE, _gold_nested_then_else(a, b), f"{tag} nested then/else")
        _check_row(got, ROW_SCALAR_BOOL, (a if flag else b).astype(np.float32), f"{tag} scalar bool")


@pytest.mark.pto
@requires_npu
def test_conditional_writes_behind_a_ub_polluter():
    """Behind a 777.0 UB polluter: a read of the destination would surface the
    polluting value on the lanes the condition does not select."""
    x, z, a, b, thr = _inputs()
    kernel = _kernel(_build_branch_shapes)
    polluter = _kernel(_build_polluter)
    gold_arm = _gold_arm_read_after_write(a)

    for r in range(3):
        polluter()
        torch.npu.synchronize()
        got = _run(kernel, x, z, a, b, thr, True)
        tag = f"polluter round {r}"
        _check_row(got, ROW_SINGLE_SIDED, _gold_single_sided(x), f"{tag} single-sided")
        _check_row(got, ROW_ARM0, gold_arm[0], f"{tag} arm row 0")
        _check_row(got, ROW_ARM1, gold_arm[1], f"{tag} arm row 1")


@pytest.mark.pto
@requires_npu
def test_carried_conditional_update_numeric():
    """A one-sided update of the carried buffer must keep the register value
    on the unselected lanes and stay ordered across iterations."""
    x, z, a, b, thr = _inputs()
    kernel = _kernel(_build_branch_shapes)
    got = _run(kernel, x, z, a, b, thr, True)
    _check_row(got, ROW_CARRIED, _gold_carried(x, thr), "carried conditional")
