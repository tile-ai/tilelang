import tilelang.language as T
from typing import Literal, Callable
from tvm.tir import IndexMap
from tilelang.intrinsics.utils import get_mma_micro_size

from tilelang.intrinsics.mfma_layout import (
    shared_16x4_to_local_64x1_layout_A,
    shared_16x16_to_local_64x4_layout_A,
    shared_16x32_to_local_64x8_layout_A,
    shared_16x64_to_local_64x16_layout_A,
)


def make_mfma_load_base_layout(
    dtype: T.dtype = T.float16, matrix: Literal["A", "B"] = "A", k_dim: int = 16, transposed: bool = False
) -> T.Fragment:
    """
    Create a layout function for storing MFMA results into a fragment buffer.
    This layout is used in conjunction with `inverse_mfma_store_layout` to
    map fragment indices to threads and local indices.

    Parameters
    ----------
    dtype : str
        The data type of the matrix.
    matrix : Literal["A", "B"]
        The mfma operand to be loaded.
    k_dim : int
        The k dimension of the mfma.
    transposed : bool
        Whether the matrix is transposed, by default False.

    Returns
    -------
    T.Fragment
        Describes how threads and indices in fragment are laid out.

    """

    assert matrix in ["A", "B"], "matrix should be either A or B"
    # s represents spatial axis
    # r represents reduction axis
    # sr represents the two dims are spatial + reduction
    # rs represents the two dims are reduction + spatial
    transform_func_sr_a: Callable = None
    transform_func_sr_b: Callable = None

    if k_dim == 4:
        transform_func_sr_a = shared_16x4_to_local_64x1_layout_A
        transform_func_sr_b = shared_16x4_to_local_64x1_layout_A
    elif k_dim == 16:
        transform_func_sr_a = shared_16x16_to_local_64x4_layout_A
        transform_func_sr_b = shared_16x16_to_local_64x4_layout_A
    elif k_dim == 32:
        transform_func_sr_a = shared_16x32_to_local_64x8_layout_A
        transform_func_sr_b = shared_16x32_to_local_64x8_layout_A
    elif k_dim == 64:
        transform_func_sr_a = shared_16x64_to_local_64x16_layout_A
        transform_func_sr_b = shared_16x64_to_local_64x16_layout_A
    else:
        raise ValueError("k_dim must be 4 or 16 or 32 or 64 currently")

    is_sr_conditions = [False]
    is_sr_conditions.append(matrix == "A" and not transposed)
    is_sr_conditions.append(matrix == "B" and transposed)
    is_sr_axis_order = any(is_sr_conditions)

    micro_size_x, micro_size_y, micro_size_k = get_mma_micro_size(dtype)

    # the layout of mma.sync is row.col.
    # so the b matrix expected a transposed basic layout
    transform_func: Callable = None
    if matrix == "A":
        transform_func = transform_func_sr_a if is_sr_axis_order else lambda i, j: transform_func_sr_a(j, i)
        micro_size_s, micro_size_r = micro_size_x, micro_size_k
    elif matrix == "B":
        transform_func = transform_func_sr_b if is_sr_axis_order else lambda i, j: transform_func_sr_b(j, i)
        micro_size_s, micro_size_r = micro_size_k, micro_size_y
    else:
        raise ValueError(f"Unsupported matrix {matrix}")

    inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

    def forward_thread(i: int, j: int) -> int:
        """
        Given the row index `i` and column index `j` in the fragment,
        """
        lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
        return lane_id

    def forward_index(i: int, j: int) -> int:
        """
        Given the row index `i` and column index `j` in the fragment,
        """
        _, local_id = inverse_mma_load_layout.map_indices([i, j])
        return local_id

    base_fragment = T.Fragment(
        [micro_size_s, micro_size_r] if is_sr_axis_order else [micro_size_r, micro_size_s],
        forward_thread_fn=forward_thread,
        forward_index_fn=forward_index,
    )
    return base_fragment


block_rows = 2
block_cols = 2
warp_rows = 2
warp_cols = 2
chunk = 2

from tilelang.tools import plot_layout

# ldmatrix layout 16x16
base_layout = make_mfma_load_base_layout(dtype=T.float16, matrix="A", transposed=False)
print(base_layout)
plot_layout(base_layout, name="base_layout")

# warp layout 32x32
warp_layout = base_layout.repeat([warp_rows, warp_cols], repeat_on_thread=False, lower_dim_first=False)
print(warp_layout)
plot_layout(warp_layout, name="warp_layout")

# block layout 64x32
block_layout = warp_layout.repeat([block_rows, 1], repeat_on_thread=True, lower_dim_first=True).replicate(block_cols)
print(block_layout)
plot_layout(block_layout, name="block_layout")
