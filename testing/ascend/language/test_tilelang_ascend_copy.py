"""Ascend copy policy spelling is normalized at the public API boundary."""

import pytest

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


def test_ascend_copy_l2_cache_ctrl_names_follow_copy_direction():
    """LD_L2CacheType (loads) and ST_L2CacheType (stores) share ints, not names."""

    @T.prim_func
    def load_path(A: T.Tensor((16,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((16,), T.float32)
            T.copy(A, ub, l2_cache_ctrl="NOTALLOC_KEEP")

    @T.prim_func
    def store_path(A: T.Tensor((16,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((16,), T.float32)
            T.copy(ub, A, l2_cache_ctrl="NOTALLOC_PW")

    # Same integers per direction, but each name only resolves on its own path.
    (load,) = calls(load_path, "tl.tileop.ascend_copy")
    (store,) = calls(store_path, "tl.tileop.ascend_copy")
    assert int(load.annotations["l2_cache_ctrl"]) == 4
    assert int(store.annotations["l2_cache_ctrl"]) == 5

    def cross_direction_load():
        @T.prim_func
        def kernel(A: T.Tensor((16,), T.float32)):
            with T.Kernel(1):
                ub = T.alloc_shared((16,), T.float32)
                T.copy(A, ub, l2_cache_ctrl="NORMAL_RED")  # store-only name

    def cross_direction_store():
        @T.prim_func
        def kernel(A: T.Tensor((16,), T.float32)):
            with T.Kernel(1):
                ub = T.alloc_shared((16,), T.float32)
                T.copy(ub, A, l2_cache_ctrl="NOTALLOC_KEEP")  # load-only name

    def invalid_name():
        @T.prim_func
        def kernel(A: T.Tensor((16,), T.float32)):
            with T.Kernel(1):
                ub = T.alloc_shared((16,), T.float32)
                T.copy(A, ub, l2_cache_ctrl="INVALID")  # codegen accepts [0, 15] only

    with pytest.raises(ValueError, match="is a store"):
        cross_direction_load()
    with pytest.raises(ValueError, match="is a load"):
        cross_direction_store()
    with pytest.raises(ValueError, match="Unknown l2_cache_ctrl"):
        invalid_name()


def test_ascend_copy_l2_cache_ctrl_passes_int_and_none_through():
    """Raw integers and None stay direction-agnostic, so the store default still works."""

    @T.prim_func
    def kernel(A: T.Tensor((16,), T.float32)):
        with T.Kernel(1):
            ub = T.alloc_shared((16,), T.float32)
            T.copy(ub, A, l2_cache_ctrl=4)
            T.copy(A, ub, l2_cache_ctrl=None)

    copies = calls(kernel, "tl.tileop.ascend_copy")
    annotated = [node for node in copies if "l2_cache_ctrl" in node.annotations]
    assert len(copies) == 2
    assert [int(node.annotations["l2_cache_ctrl"]) for node in annotated] == [4]
