"""Device numeric tests for runtime trip counts in a preserved loop shell.

A ``T.serial`` loop inside ``T.Parallel`` is copied through as a scalar shell,
so its trip count does not have to be known at compile time. These tests pin
the runtime behaviour, including the two boundaries that only show up when the
count is dynamic:

* ``iters == 0``: the shell body never runs, so the output stays at the seed
  value (nothing inside the shell can produce one);
* ``iters > 0``: the carrier stays in UB and every serial iteration reads and
  writes it through load/store; the passes preserve the shell and convert its
  body only (no carry-promotion), so per-iteration ordering relies on the
  backend's vecscope mem_bar.

The carried and the carry-free shells run in one kernel (row 0 is the carried
one, row 1 the plain one), so one device compile pins both.
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


@lru_cache(maxsize=None)
def _compile_builder(builder):
    """Compile each builder once: a device compile dominates the test cost."""
    return tilelang.compile(builder(), target="pto")


def _build_runtime_shells():
    """Row 0: seed 0 + runtime-bound accumulate (``y = iters * x``).

    Row 1: no carry — the shell only writes ``2 * x``, and with ``iters == 0``
    the output stays at the seed (-7).
    """

    @T.prim_func
    def main(X: T.Tensor((E,), "float32"), Y: T.Tensor((2, E), "float32"), iters: T.int32):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((2, E), "float32")
            T.copy(X, x_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.Parallel(E):
                    y_ub[0, i] = T.float32(0.0)
                    for _t in T.serial(iters):
                        y_ub[0, i] = y_ub[0, i] + x_ub[i]
                # Seed the plain row first: with ``iters == 0`` the output must
                # be this seed (the loop body is not a carry, so nothing else
                # can produce a value).
                for i in T.Parallel(E):
                    y_ub[1, i] = T.float32(-7.0)
                for i in T.Parallel(E):
                    for _t in T.serial(iters):
                        y_ub[1, i] = x_ub[i] * T.float32(2.0)
            T.copy(y_ub, Y)

    return main


def _build_step_two_shell():
    """seed 0 + ``T.serial(0, 4, 2)``: two iterations, so ``y = 2 * x``."""

    @T.prim_func
    def main(X: T.Tensor((E,), "float32"), Y: T.Tensor((E,), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((E,), "float32")
            y_ub = T.alloc_shared((E,), "float32")
            T.copy(X, x_ub)
            with T.SimdVF(lanes=LANES):
                for i in T.Parallel(E):
                    y_ub[i] = T.float32(0.0)
                    for _t in T.serial(0, 4, 2):
                        y_ub[i] = y_ub[i] + x_ub[i]
            T.copy(y_ub, Y)

    return main


@pytest.mark.pto
@requires_npu
@pytest.mark.parametrize("iters", [0, 1, 4])
def test_carried_runtime_shell_counts_every_iteration(iters):
    rng = np.random.default_rng(11)
    x = rng.uniform(-2.0, 2.0, E).astype(np.float32)
    # The kernel takes a scalar argument, so the output cannot be derived from
    # ``out_idx`` (a scalar parameter is not a tensor): pass the buffer in.
    kernel = _compile_builder(_build_runtime_shells)
    out = torch.empty((2, E), dtype=torch.float32, device=_dev())
    kernel(torch.from_numpy(x).to(_dev()), out, iters)
    got = out.cpu().numpy().astype(np.float32)[0]
    want = (x * np.float32(iters)).astype(np.float32)
    bad = int((~np.isclose(got, want, rtol=1e-5, atol=1e-6)).sum())
    assert bad == 0, f"carried iters={iters}: {bad}/{E} mismatched"


@pytest.mark.pto
@requires_npu
def test_step_two_shell_runs_two_iterations():
    rng = np.random.default_rng(13)
    x = rng.uniform(-2.0, 2.0, E).astype(np.float32)
    kernel = _compile_builder(_build_step_two_shell)
    out = torch.empty(E, dtype=torch.float32, device=_dev())
    kernel(torch.from_numpy(x).to(_dev()), out)
    got = out.cpu().numpy().astype(np.float32)
    want = (x * np.float32(2.0)).astype(np.float32)
    bad = int((~np.isclose(got, want, rtol=1e-5, atol=1e-6)).sum())
    assert bad == 0, f"step=2 must run exactly two iterations: {bad}/{E} mismatched"


@pytest.mark.pto
@requires_npu
@pytest.mark.parametrize("iters", [0, 3])
def test_plain_runtime_shell_runs_or_skips(iters):
    rng = np.random.default_rng(12)
    x = rng.uniform(-2.0, 2.0, E).astype(np.float32)
    kernel = _compile_builder(_build_runtime_shells)
    out = torch.full((2, E), -7.0, dtype=torch.float32, device=_dev())
    kernel(torch.from_numpy(x).to(_dev()), out, iters)
    got = out.cpu().numpy().astype(np.float32)[1]
    if iters == 0:
        # The shell never runs, so the store inside it never executes and the
        # output keeps the seed: pin that (no phantom write).
        assert np.all(got == np.float32(-7.0)), f"iters=0 must not write, got {got[:4]}"
        return
    want = (x * np.float32(2.0)).astype(np.float32)
    bad = int((~np.isclose(got, want, rtol=1e-5, atol=1e-6)).sum())
    assert bad == 0, f"plain iters={iters}: {bad}/{E} mismatched"
