"""Operand and scale-factor movement for BF16 and MXFP8 kernels."""

import tilelang.ascend.language as T

from ..config import Major
from .common import matrix_region


def make_gm_to_l1_copy(major, l2_cache_ctrl):
    """Preserve the input's physical layout in L1; transpose only when loading L0."""

    @T.macro
    def copy_operand(gm, l1, mn_idx, k_idx, actual_mn, actual_k, batch_idx=0):
        if major == Major.K:
            T.copy(
                matrix_region(gm, mn_idx, k_idx, actual_mn, actual_k, batch_idx),
                l1[0:actual_mn, 0:actual_k],
                l2_cache_ctrl=l2_cache_ctrl,
            )
        else:
            T.copy(
                matrix_region(gm, k_idx, mn_idx, actual_k, actual_mn, batch_idx),
                l1[0:actual_k, 0:actual_mn],
                l2_cache_ctrl=l2_cache_ctrl,
            )

    return copy_operand


def make_l1_to_l0_copy(major):
    """Load data and optional MX scales as separate copy operations."""

    @T.macro
    def copy_operand(l1, l0_tile, mn_idx, k_idx, actual_mn, actual_k, l1_scale=None, l0_scale=None):
        if major == Major.K:
            T.copy(l1[mn_idx : mn_idx + actual_mn, k_idx : k_idx + actual_k], l0_tile)
        else:
            T.copy(l1[k_idx : k_idx + actual_k, mn_idx : mn_idx + actual_mn], l0_tile, transpose=True)
        if l1_scale is not None:
            T.copy(l1_scale, l0_scale)

    return copy_operand


def make_scale_copy(l2_cache_ctrl):
    """Pack external [SF pair, MN] scales into the L1 [MN, SF pair] layout."""

    @T.macro
    def copy_scale(gm, l1, mn_idx, sf_idx, actual_mn, sf_pairs, batch_idx=0):
        T.copy(
            matrix_region(gm, sf_idx, mn_idx, sf_pairs, actual_mn, batch_idx),
            l1[0:actual_mn, 0:sf_pairs],
            transpose=True,
            l2_cache_ctrl=l2_cache_ctrl,
        )

    return copy_scale
