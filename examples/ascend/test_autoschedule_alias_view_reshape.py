"""Regression coverage for auto-schedule dependencies across buffer aliases."""

import pytest
import torch
import tilelang
from tilelang import language as T


TILE = 1024
N_CORES = 4
N = TILE * N_CORES


def _alias_kernel(use_view: bool):
    @T.prim_func
    def main(A: T.Tensor((N,), T.float32), O: T.Tensor((N,), T.float32)):
        with T.Kernel(N_CORES) as bx:
            ub = T.alloc_shared((TILE,), T.float32)
            alias = T.view(ub, (TILE // 2, 2)) if use_view else T.reshape(ub, (TILE // 2, 2))

            for tile in T.Pipelined(TILE, num_stages=2):
                idx = bx * TILE + tile
                ub[tile] = A[idx]
                O[idx] = alias[tile // 2, tile % 2] + T.float32(1.0)

    return main


def _run_alias(use_view: bool, target: str):
    kernel = tilelang.compile(_alias_kernel(use_view), target=target)
    device = torch.device("npu")
    src = torch.randn(N, dtype=torch.float32, device=device)
    out = torch.empty(N, dtype=torch.float32, device=device)
    kernel(src, out)
    torch.npu.synchronize()
    expected = src + 1.0
    max_diff = (out - expected).abs().max().item()
    assert max_diff < 1e-6, f"max_diff={max_diff:.2e}"


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_autoschedule_reshape_alias_dependency(target):
    _run_alias(use_view=False, target=target)


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_autoschedule_view_alias_dependency(target):
    _run_alias(use_view=True, target=target)


if __name__ == "__main__":
    for target in ("ascend", "pto"):
        test_autoschedule_reshape_alias_dependency(target)
        print(f"PASS: test_autoschedule_reshape_alias_dependency ({target})")
        test_autoschedule_view_alias_dependency(target)
        print(f"PASS: test_autoschedule_view_alias_dependency ({target})")
