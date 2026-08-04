__all__ = [
    "dim3",
    "ThreadIdx",
    "BlockIdx",
    "GridDim",
    "rasterization2DRow",
    "rasterization2DColumn",
    "rasterization2DRowWithCluster",
    "rasterization2DColumnWithCluster",
]

import cutlass.cute as cute

try:
    from cutlass import Constexpr
except ImportError:
    from cutlass.cute.typing import Constexpr
from dataclasses import dataclass


@dataclass(frozen=True)
class dim3:
    """Three-dimensional CUDA index tuple."""

    x: int
    y: int
    z: int


def ThreadIdx() -> dim3:
    """Return the current CUDA thread index."""

    return dim3(*cute.arch.thread_idx())


def BlockIdx() -> dim3:
    """Return the current CUDA block index."""

    return dim3(*cute.arch.block_idx())


def GridDim() -> dim3:
    """Return the CUDA grid dimensions."""

    return dim3(*cute.arch.grid_dim())


@cute.jit
def rasterization2DRow(panel_width: Constexpr[int]) -> dim3:
    """Map block indices to row-major swizzled rasterization coordinates."""

    blockIdx = BlockIdx()
    gridDim = GridDim()
    block_idx = blockIdx.x + blockIdx.y * gridDim.x
    grid_size = gridDim.x * gridDim.y
    panel_size = panel_width * gridDim.x
    panel_offset = block_idx % panel_size
    panel_idx = block_idx // panel_size
    total_panel = cute.ceil_div(grid_size, panel_size)
    stride = panel_width if panel_idx + 1 < total_panel else (grid_size - panel_idx * panel_size) // gridDim.x
    col_idx = (gridDim.x - 1 - panel_offset // stride) if (panel_idx & 1 != 0) else (panel_offset // stride)
    row_idx = panel_offset % stride + panel_idx * panel_width
    return dim3(col_idx, row_idx, blockIdx.z)


@cute.jit
def rasterization2DColumn(panel_width: Constexpr[int]) -> dim3:
    """Map block indices to column-major swizzled rasterization coordinates."""

    blockIdx = BlockIdx()
    gridDim = GridDim()
    block_idx = blockIdx.x + blockIdx.y * gridDim.x
    grid_size = gridDim.x * gridDim.y
    panel_size = panel_width * gridDim.y
    panel_offset = block_idx % panel_size
    panel_idx = block_idx // panel_size
    total_panel = cute.ceil_div(grid_size, panel_size)
    stride = panel_width if panel_idx + 1 < total_panel else (grid_size - panel_idx * panel_size) // gridDim.y
    row_idx = (gridDim.y - 1 - panel_offset // stride) if (panel_idx & 1 != 0) else (panel_offset // stride)
    col_idx = panel_offset % stride + panel_idx * panel_width
    return dim3(col_idx, row_idx, blockIdx.z)


@cute.jit
def rasterization2DRowWithCluster(panel_width: Constexpr[int], cluster_dim_x: Constexpr[int]) -> dim3:
    """Row-major swizzle at cluster granularity while preserving intra-cluster x rank."""

    blockIdx = BlockIdx()
    gridDim = GridDim()
    num_cluster_x = gridDim.x // cluster_dim_x
    intra_cluster_x = blockIdx.x % cluster_dim_x
    cluster_x = blockIdx.x // cluster_dim_x
    cluster_idx = cluster_x + blockIdx.y * num_cluster_x
    cluster_grid_size = num_cluster_x * gridDim.y
    panel_size = panel_width * num_cluster_x
    panel_offset = cluster_idx % panel_size
    panel_idx = cluster_idx // panel_size
    total_panel = cute.ceil_div(cluster_grid_size, panel_size)
    stride = panel_width if panel_idx + 1 < total_panel else (cluster_grid_size - panel_idx * panel_size) // num_cluster_x
    swizzled_cluster_x = (
        (num_cluster_x - 1 - panel_offset // stride) if (panel_idx & 1 != 0) else (panel_offset // stride)
    )
    swizzled_cluster_y = panel_offset % stride + panel_idx * panel_width
    col_idx = swizzled_cluster_x * cluster_dim_x + intra_cluster_x
    return dim3(col_idx, swizzled_cluster_y, blockIdx.z)


@cute.jit
def rasterization2DColumnWithCluster(panel_width: Constexpr[int], cluster_dim_x: Constexpr[int]) -> dim3:
    """Column-major swizzle at cluster granularity while preserving intra-cluster x rank."""

    blockIdx = BlockIdx()
    gridDim = GridDim()
    num_cluster_x = gridDim.x // cluster_dim_x
    intra_cluster_x = blockIdx.x % cluster_dim_x
    cluster_x = blockIdx.x // cluster_dim_x
    cluster_idx = cluster_x + blockIdx.y * num_cluster_x
    cluster_grid_size = num_cluster_x * gridDim.y
    panel_size = panel_width * gridDim.y
    panel_offset = cluster_idx % panel_size
    panel_idx = cluster_idx // panel_size
    total_panel = cute.ceil_div(cluster_grid_size, panel_size)
    stride = panel_width if panel_idx + 1 < total_panel else (cluster_grid_size - panel_idx * panel_size) // gridDim.y
    swizzled_cluster_y = (gridDim.y - 1 - panel_offset // stride) if (panel_idx & 1 != 0) else (panel_offset // stride)
    swizzled_cluster_x = panel_offset % stride + panel_idx * panel_width
    col_idx = swizzled_cluster_x * cluster_dim_x + intra_cluster_x
    return dim3(col_idx, swizzled_cluster_y, blockIdx.z)
