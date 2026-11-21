"""The m-grouped scheduler must not drop tiles for groups inside an alignment gap.

``AscendMGroupedTileScheduler`` places each group at the next ``alignment``
boundary and hands out one m-block per tile. A group whose row range ends at or
below that boundary owns no rows: it must contribute zero tiles and leave the
running block count untouched.

Before this test the empty group decremented the count (``ceildiv`` of a negative
row deficit), so every later group landed at the wrong global position and a whole
tile was silently skipped -- the affected rows were never computed, with no error.

No existing test drives a grouped scheduler: the files under
``testing/ascend/auto_schedule`` cover the generic scheduling passes, so this is a
new file rather than an extension. It is the only case that covers a non-empty
layout with a group falling entirely inside an alignment gap.
"""

import pytest
import torch

import tilelang
import tilelang.ascend.language as T
import tilelang.testing

BLOCK_M = 128
ALIGNMENT = 256


def _dispatch_kernel(psum, slots):
    """Record ``(group_idx, m_idx)`` for every tile the scheduler hands out."""
    num_groups = len(psum)
    shape_m = psum[-1]

    @T.prim_func
    def main(layout: T.Tensor((num_groups,), "int32"), groups: T.Tensor((slots,), "int32"), blocks: T.Tensor((slots,), "int32")):
        with T.Kernel(1) as bx:
            sched = T.AscendMGroupedTileScheduler(
                block_m=BLOCK_M,
                block_n=BLOCK_M,
                num_cores=1,
                shape_m=shape_m,
                shape_n=BLOCK_M,
                grouped_layout=layout,
                num_groups=num_groups,
                alignment=ALIGNMENT,
            )
            sched.init(bx)
            it = T.alloc_var("int32", init=0)
            while sched.valid():
                groups[it] = sched.get_group_idx()
                blocks[it] = sched.m_idx
                it = it + 1
                sched.next_block()

    return main


def _expected(psum):
    """(group, global m-block) pairs a complete pass must produce."""
    expected = []
    previous = 0
    for group, end in enumerate(psum):
        start = 0 if group == 0 else ((previous + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        rows = end - start
        if rows > 0:
            base = start // BLOCK_M
            for offset in range((rows + BLOCK_M - 1) // BLOCK_M):
                expected.append((group, base + offset))
        previous = end
    return expected


@pytest.mark.parametrize(
    "psum",
    [
        # control: no gap in use
        [128, 256],
        # control: a real gap, but every group still owns rows
        [100, 400],
        [100, 200, 400],
        # regression: group 1 ends 146 rows below align(100) = 256, owning no rows
        [100, 110, 400],
        # regression: same shape at a smaller scale
        [64, 96, 400],
        # regression: the empty group is the last one
        [300, 320, 640],
    ],
)
@tilelang.testing.requires_ascend
def test_group_inside_alignment_gap_is_skipped(psum):
    expected = _expected(psum)
    slots = len(expected) + 4

    kernel = tilelang.compile(_dispatch_kernel(psum, slots), target="ascend")
    layout = torch.tensor(psum, device="npu", dtype=torch.int32)
    groups = torch.full((slots,), -1, device="npu", dtype=torch.int32)
    blocks = torch.full((slots,), -1, device="npu", dtype=torch.int32)
    kernel(layout, groups, blocks)
    torch.npu.synchronize()

    dispatched = [(int(g), int(b)) for g, b in zip(groups.cpu().tolist(), blocks.cpu().tolist()) if g >= 0 and b >= 0]

    # Every tile the layout describes must be dispatched exactly once.
    assert sorted(dispatched) == expected
    assert len(dispatched) == len(set(dispatched))


if __name__ == "__main__":
    tilelang.testing.main()
