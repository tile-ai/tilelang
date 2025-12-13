import cutlass.cute as cute
from cutlass.cute.typing import Constexpr
from dataclasses import dataclass


@dataclass(frozen=True)
class dim3:
    x: int
    y: int
    z: int


def ThreadIdx() -> dim3:
    return dim3(*cute.arch.thread_idx())


def BlockIdx() -> dim3:
    return dim3(*cute.arch.block_idx())


def GridDim() -> dim3:
    return dim3(*cute.arch.grid_dim())


@cute.jit
def rasterization2DRow(panel_width: Constexpr[int]) -> dim3:
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
