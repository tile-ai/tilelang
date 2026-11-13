"""Ascend copy policy spelling is normalized at the public API boundary."""

from tilelang.ascend import language as T
from testing.ascend._ir import calls


def test_ascend_copy_l2_cache_ctrl_string_annotation_matches_keyword():
    """The Ascend dialect owns l2_cache_ctrl; names normalize to ints.

    The keyword and the annotation must agree, and a string policy name must
    reach the tile op as the integer the backend expects (here NORMAL_FV -> 0).
    """

    @T.prim_func
    def annotation_path(A: T.Tensor((16,), T.float32), B: T.Tensor((16,), T.float32)):
        with T.Kernel(1):
            T.copy(A, B, annotations={"l2_cache_ctrl": "NORMAL_FV"})

    @T.prim_func
    def keyword_path(A: T.Tensor((16,), T.float32), B: T.Tensor((16,), T.float32)):
        with T.Kernel(1):
            T.copy(A, B, l2_cache_ctrl="NORMAL_FV")

    for func in (annotation_path, keyword_path):
        (copy,) = calls(func, "tl.tileop.ascend_copy")
        assert int(copy.annotations["l2_cache_ctrl"]) == 0


def test_pad_value_rejects_l1_destination():
    """Only the GM->UB path reads the pad register; L1 padding accepts zero only."""
    import pytest

    with pytest.raises(ValueError, match="UB buffer"):

        @T.prim_func
        def main(A: T.Tensor((64, 64), "float16"), B: T.Tensor((64, 64), "float16")):
            with T.Kernel(1):
                a = T.alloc_l1((64, 64), "float16")
                ub = T.alloc_shared((64, 64), "float16")
                T.copy(A, a, pad_value=0.0)
                T.copy(a, ub)
                T.copy(ub, B)
