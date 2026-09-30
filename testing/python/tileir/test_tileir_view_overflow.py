"""T.view must accept tensors whose bit count reaches or exceeds 2**31.

bits_product used to multiply extents as int32 PrimExprs, wrapping at 2**31 bits
(256 MiB), and prim_expr_equal built an int32 IntImm from the result, so a view of
any tensor >= 256 MiB was rejected before T.view's can_prove_equal fallback ran.
The 256 MiB case is KV [32, 8192, 512] bf16 -- exactly 2**31 bits.  Construction
only; no compilation or GPU is needed to hit the check.
"""

import pytest

import tilelang.language as T


def _build(batch: int, seq_len_kv: int, dim: int):
    @T.prim_func
    def main(KV: T.Tensor([batch, seq_len_kv, dim], T.bfloat16)):
        with T.Kernel(1, threads=128):
            packed = T.view(KV, [batch, seq_len_kv, dim // 4], T.int64)
            staging = T.alloc_shared([1, dim // 4], T.int64)
            T.copy(packed[0, 0:1, :], staging)

    return main


@pytest.mark.parametrize(
    "batch",
    [
        pytest.param(16, id="128MiB-below-2^31-bits"),
        pytest.param(32, id="256MiB-exactly-2^31-bits"),
        pytest.param(64, id="512MiB-above-2^31-bits"),
    ],
)
def test_view_accepts_tensors_at_and_above_2_31_bits(batch: int) -> None:
    # Fails on the unfixed tree with "Literal value 2147483648 exceeds" once the
    # tensor reaches 2**31 bits; must construct cleanly at every size.
    _build(batch, 8192, 512)


def test_view_still_rejects_mismatched_capacity() -> None:
    # The fix must not weaken the check itself: a view that changes the total bit
    # count is still an error.
    with pytest.raises(AssertionError, match="shape check failed"):

        @T.prim_func
        def main(KV: T.Tensor([32, 8192, 512], T.bfloat16)):
            with T.Kernel(1, threads=128):
                T.view(KV, [32, 8192, 100], T.int64)
