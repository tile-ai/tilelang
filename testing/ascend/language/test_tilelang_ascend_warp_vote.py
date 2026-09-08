import torch
import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


@T.prim_func
def warp_vote_kernel(
    sync_mask: T.Tensor((1,), "uint32"),
    ballot_out: T.Tensor((32,), "uint32"),
    masked_ballot_out: T.Tensor((32,), "uint32"),
    active_out: T.Tensor((32,), "uint32"),
    sync_out: T.Tensor((32,), "uint32"),
    masked_sync_out: T.Tensor((32,), "uint32"),
):
    with T.Kernel(1) as _:
        sync_buffer = T.alloc_shared((32,), "uint32")
        masked_sync_buffer = T.alloc_shared((32,), "uint32")
        with T.SimtVF(threads=32):
            tx = T.get_thread_binding(0)
            active = T.activemask()
            ballot = T.ballot(tx < 16)
            masked_ballot = T.ballot_sync(tx < 16, sync_mask[0])

            sync_buffer[tx] = T.cast(tx + 1, "uint32")
            T.sync_warp()
            sync_out[tx] = sync_buffer[tx ^ 1]

            is_member = ((sync_mask[0] >> tx) & 1) != 0
            if is_member:
                masked_sync_buffer[tx] = T.cast(tx + 101, "uint32")
            T.sync_warp(sync_mask[0])
            if is_member:
                masked_sync_out[tx] = masked_sync_buffer[tx ^ 2]
            else:
                masked_sync_out[tx] = 0

            ballot_out[tx] = T.cast(ballot, "uint32")
            masked_ballot_out[tx] = T.cast(masked_ballot, "uint32")
            active_out[tx] = T.cast(active, "uint32")


@pytest.mark.parametrize("mask", [0x55555555, 0xAAAAAAAA], ids=["even_lanes", "odd_lanes"])
def test_ascend_warp_vote_correctness(mask):
    kernel = tilelang.compile(
        warp_vote_kernel,
        out_idx=[1, 2, 3, 4, 5],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
    )
    assert "#include <simt_api/cooperative_groups.h>" in kernel.get_kernel_source()
    sync_mask = torch.tensor([mask], dtype=torch.uint32, device="npu")
    ballot_out, masked_ballot_out, active_out, sync_out, masked_sync_out = kernel(sync_mask)
    torch.npu.synchronize()
    torch.testing.assert_close(ballot_out.cpu(), torch.full((32,), 0xFFFF, dtype=torch.uint32))
    torch.testing.assert_close(masked_ballot_out.cpu(), torch.full((32,), mask & 0xFFFF, dtype=torch.uint32))
    torch.testing.assert_close(active_out.cpu(), torch.full((32,), 0xFFFFFFFF, dtype=torch.uint32))
    expected_sync = torch.tensor([(lane ^ 1) + 1 for lane in range(32)], dtype=torch.uint32)
    torch.testing.assert_close(sync_out.cpu(), expected_sync)
    expected_masked_sync = torch.tensor(
        [(lane ^ 2) + 101 if mask & (1 << lane) else 0 for lane in range(32)],
        dtype=torch.uint32,
    )
    torch.testing.assert_close(masked_sync_out.cpu(), expected_masked_sync)


if __name__ == "__main__":
    tilelang.testing.main()
