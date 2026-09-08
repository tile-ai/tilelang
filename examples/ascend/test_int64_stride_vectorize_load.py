"""pytest test for example_int64_stride_vectorize_load.py."""

import pytest
import torch
import tilelang

from example_int64_stride_vectorize_load import DIM, repro_kernel


@pytest.mark.parametrize("target", ["ascend", pytest.param("pto", marks=pytest.mark.pto)])
def test_int64_stride_vectorize_load(target):
    device = torch.device("npu")

    num_slots, cache_size, total_c = 2, 32, 4
    state_cache = torch.randn(num_slots, cache_size, DIM, dtype=torch.float32, device=device)
    slot_idx = torch.zeros(1, dtype=torch.int32, device=device)
    out = torch.zeros(total_c, DIM, dtype=torch.float32, device=device)

    kernel = tilelang.compile(repro_kernel(), target=target)

    kernel(state_cache, slot_idx, out)
    torch.npu.synchronize()

    # Kernel copies state_cache[slot_idx[0], 0, :] into out[0, :].
    assert torch.equal(out[0], state_cache[int(slot_idx[0]), 0])


if __name__ == "__main__":
    test_int64_stride_vectorize_load(target="ascend")
    print("PASS: test_int64_stride_vectorize_load (ascend)")
    test_int64_stride_vectorize_load(target="pto")
    print("PASS: test_int64_stride_vectorize_load (pto)")
