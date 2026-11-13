"""occupancy_report: blocks per compute unit from clang's resource usage remarks."""

import pytest
from tilelang.contrib.hip_resource_info import KernelResourceUsage, occupancy_report


def _usage(occupancy, scratch=0):
    return KernelResourceUsage(scratch_bytes=scratch, extra={"Occupancy [waves/SIMD]": str(occupancy)})


@pytest.mark.parametrize(
    ("occupancy", "threads", "lds_bytes", "by_occupancy", "by_lds", "blocks"),
    [
        (1, 256, 24576, 1, 2, 1),  # 256x128x32 without the attribute: occupancy binds
        (3, 256, 24576, 3, 2, 2),  # with it: LDS binds
        (3, 768, 0, 1, None, 1),
        (4, 512, 21000, 2, 3, 2),
        (2, 128, 16384, 4, 4, 4),
        (4, 256, 21760, 4, 2, 2),  # 21760 bytes take 43 units of 512: 22016 allocated, so 2 blocks, not 3
    ],
)
def test_blocks_per_cu(occupancy, threads, lds_bytes, by_occupancy, by_lds, blocks):
    report = occupancy_report(_usage(occupancy), threads, lds_bytes)
    assert (report.blocks_by_occupancy, report.blocks_by_lds, report.blocks_per_cu) == (by_occupancy, by_lds, blocks)


def test_warnings():
    assert "occupancy limits" in occupancy_report(_usage(1), 256, 24576).warnings[0]
    assert "scratch" in occupancy_report(_usage(3, scratch=184), 256).warnings[0]
    assert occupancy_report(_usage(3), 256, 24576).warnings == []
