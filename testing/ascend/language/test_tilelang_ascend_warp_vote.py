import torch
import pytest

import tilelang
import tilelang.ascend.language as T
import tilelang.testing
from tilelang.engine.lower import lower


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


@T.prim_func
def warp_vote_any_all_kernel(
    vote_mask: T.Tensor((1,), "uint32"),
    any_out: T.Tensor((32,), "int32"),
    all_out: T.Tensor((32,), "int32"),
):
    with T.Kernel(1) as _, T.SimtVF(threads=32):
        tx = T.get_thread_binding(0)
        any_out[tx] = T.any_sync(tx < 8, vote_mask[0])
        all_out[tx] = T.all_sync(tx < 8, vote_mask[0])


def test_ascend_any_all_sync_lower_to_masked_ballot():
    """CANN's asc_any/asc_all take no lane mask, so both votes are derived from
    asc_ballot. Before this, neither op had an AscendC arm and lowering aborted
    with an unresolved tl.any_sync / tl.all_sync call."""
    source = lower(warp_vote_any_all_kernel, target="ascend").kernel_source
    assert "asc_ballot(" in source
    # ANY: some lane in the mask voted true.
    assert ") != 0U))" in source
    # ALL: no lane in the mask voted false -- note the negated predicate.
    assert "asc_ballot(!(" in source
    assert ") == 0U))" in source
    assert "tl.any_sync" not in source
    assert "tl.all_sync" not in source


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    ("mask", "expected_any", "expected_all"),
    [
        # ballot(tx < 8) == 0x000000FF.
        (0x0000000F, 1, 1),  # every masked lane votes true
        (0x000000FF, 1, 1),  # exactly the voting lanes
        (0x0000FFFF, 1, 0),  # lanes 8..15 vote false
        (0x00FF0000, 0, 0),  # no masked lane votes true
    ],
    ids=["subset", "exact", "partial", "disjoint"],
)
def test_ascend_any_all_sync_honours_lane_mask(mask, expected_any, expected_all):
    """The mask operand selects the voting lanes, matching ballot_sync."""
    kernel = tilelang.compile(
        warp_vote_any_all_kernel,
        out_idx=[1, 2],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False,
            tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        },
    )
    vote_mask = torch.tensor([mask], dtype=torch.uint32, device="npu")
    any_out, all_out = kernel(vote_mask)
    torch.npu.synchronize()
    torch.testing.assert_close(any_out.cpu(), torch.full((32,), expected_any, dtype=torch.int32))
    torch.testing.assert_close(all_out.cpu(), torch.full((32,), expected_all, dtype=torch.int32))


if __name__ == "__main__":
    tilelang.testing.main()
