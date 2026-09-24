from ..scan_tiling_data import ScanStrategy

STATUS = "DESIGN_ONLY"
STRATEGY = ScanStrategy(
    "core_partition",
    "multiple cores own contiguous partitions of one long row",
    False,
    "three kernels",
    "partial scan, ordered partition-prefix scan, then offset add; count all launches and workspace",
)


def workspace_elements(rows: int, partitions_per_row: int) -> int:
    return rows * partitions_per_row
