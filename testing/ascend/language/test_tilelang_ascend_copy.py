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


def test_l0c_dyn_scope_survives_copy_check():
    """`shared.l0c` and `shared.l0c.dyn` name the same hardware scope."""
    from tilelang.engine.lower import lower

    @T.prim_func
    def main(A: T.Tensor((64, 64), "float32"), C: T.Tensor((64, 64), "float16")):
        with T.Kernel(1):
            res = T.alloc_l0c((64, 64), "float32", scope="shared.l0c.dyn")
            a = T.alloc_l1((64, 64), "float32")
            T.copy(A, a)
            T.gemm(a, a, res, transpose_B=True, clear_accum=True)
            T.copy(res, C)

    lower(main, target="ascend")
