"""Test for a Let that reads a buffer mutated later in the same pipelined loop.

The accumulator `buf` is read into `a` BEFORE the inner accumulation loop, so
`ub[w]` must be the exclusive prefix sum (value of `buf` from before iteration
w's adds). Under auto-schedule, the Let is cloned into the consumer's pipeline
stage; without the stage-order dependency the clone reads `buf` one iteration
too late and produces the inclusive prefix instead. This test guards that fix.
"""

import pytest
import torch
import tilelang
from tilelang import language as T

TILE_N = 8192
N_CORES = 64
N = TILE_N * N_CORES
NUM_STAGES = 2


@tilelang.jit(out_idx=-1)
def _let_stage_kernel():
    @T.prim_func
    def main(
        gm: T.Tensor((N,), T.float32),
        out: T.Tensor((N_CORES, TILE_N), T.float32),
    ):
        with T.Kernel(N_CORES) as core_id:
            buf = T.alloc_var(T.float32)
            ub = T.alloc_shared((TILE_N,), T.float32)
            buf = 0
            for w in T.Pipelined(TILE_N, num_stages=NUM_STAGES):
                x = w * N_CORES + core_id
                a = buf  # read buf BEFORE this iteration's adds
                for _ in range(10):
                    buf += gm[x]
                ub[w] = a
            T.copy(ub, out[core_id, :])

    return main


def ref_program(gm: torch.Tensor) -> torch.Tensor:
    # gm is laid out as [w, core] with core stride = 1: gm[w * N_CORES + core].
    g = gm.reshape(TILE_N, N_CORES)
    contrib = 10.0 * g
    exclusive = torch.cumsum(contrib, dim=0) - contrib  # [w, core]
    return exclusive.transpose(0, 1).contiguous()  # [core, w]


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_let_stage_dependency(target):
    device = torch.device("npu")
    torch.manual_seed(0)
    gm = torch.randn(N, dtype=torch.float32, device=device)
    prim = _let_stage_kernel.get_tir()
    kernel = tilelang.compile(prim, target=target, out_idx=-1)
    out = kernel(gm)
    torch.npu.synchronize()
    expected = ref_program(gm)
    max_diff = (out - expected).abs().max().item()
    print(f"max_diff={max_diff:.2e}")
    assert max_diff < 1e-1, f"max_diff={max_diff:.2e}"


if __name__ == "__main__":
    for target in ("ascend", "pto"):
        test_let_stage_dependency(target)
        print(f"PASS ({target})")
