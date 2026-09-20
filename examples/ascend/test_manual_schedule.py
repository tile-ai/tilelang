"""Correctness test for example_manual_schedule.py."""

import pytest
import torch
import tilelang

from example_manual_schedule import NUM_BLOCKS, TILE_ELEMS, manual_schedule_vector_add, ref_program


TARGETS = ["ascend", pytest.param("pto", marks=pytest.mark.pto)]


@pytest.mark.parametrize("target", TARGETS)
def test_manual_schedule_vector_add(target):
    num_tiles = 4
    n = NUM_BLOCKS * TILE_ELEMS * num_tiles
    kernel = tilelang.compile(
        manual_schedule_vector_add(num_tiles),
        target=target,
        out_idx=-1,
    )

    device = torch.device("npu")
    torch.manual_seed(0)
    a = torch.randn(n, dtype=torch.float32, device=device)
    b = torch.randn(n, dtype=torch.float32, device=device)
    result = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(result, ref_program(a, b), rtol=0, atol=0)


if __name__ == "__main__":
    test_manual_schedule_vector_add()
    print("PASS: test_manual_schedule_vector_add")
