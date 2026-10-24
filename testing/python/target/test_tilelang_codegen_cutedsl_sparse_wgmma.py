"""CuTeDSL sparse WGMMA: the trailing immediates depend on the dtype form.

Only the f16/bf16 grammars of ``wgmma.mma_async.sp`` take transpose immediates;
tf32, fp8 and the integer forms take the two scale immediates only. Getting this
wrong is a ``ptxas`` error rather than a wrong result, so pin the grammar here.
"""

import pytest

import tilelang.testing

pytestmark = tilelang.testing.requires_cuda.marks()


def _tail(*args):
    from tilelang.contrib.cutedsl.gemm_v2_sp import _sparse_tail

    return _sparse_tail(*args)


@pytest.mark.parametrize("ptx_a", ["f16", "bf16"])
def test_f16_bf16_take_transpose_immediates(ptx_a):
    assert _tail(ptx_a, 0, 1, 1, True, False) == "p, 1, 1, 1, 0;"
    assert _tail(ptx_a, 0, 1, -1, False, True) == "p, 1, -1, 0, 1;"
    # Register A is always K-major, so the RS form drops trans-a.
    assert _tail(ptx_a, 4, 1, 1, True, True) == "p, 1, 1, 1;"


@pytest.mark.parametrize("ptx_a", ["tf32", "e4m3", "e5m2", "s8", "u8"])
def test_other_forms_take_scale_only(ptx_a):
    assert _tail(ptx_a, 0, 1, 1, True, True) == "p, 1, 1;"
    assert _tail(ptx_a, 4, -1, 1, True, True) == "p, -1, 1;"


if __name__ == "__main__":
    tilelang.testing.main()
