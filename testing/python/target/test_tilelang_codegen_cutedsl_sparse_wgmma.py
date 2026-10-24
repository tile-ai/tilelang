"""CuTeDSL WGMMA (sparse and dense RS): the trailing immediates depend on the dtype form.

Only the f16/bf16 grammars of ``wgmma.mma_async.sp`` take transpose immediates;
tf32 and fp8 take the two scale immediates only, and the integer forms only the
scale-d predicate. Getting this wrong is a ``ptxas`` error rather than a wrong
result, so pin the grammar here.
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


@pytest.mark.parametrize("ptx_a", ["tf32", "e4m3", "e5m2"])
def test_other_forms_take_scale_only(ptx_a):
    assert _tail(ptx_a, 0, 1, 1, True, True) == "p, 1, 1;"
    assert _tail(ptx_a, 4, -1, 1, True, True) == "p, -1, 1;"


@pytest.mark.parametrize("ptx_a", ["s8", "u8"])
def test_integer_forms_take_scale_d_only(ptx_a):
    assert _tail(ptx_a, 0, 1, 1, True, True) == "p;"
    assert _tail(ptx_a, 4, 1, 1, False, False) == "p;"


@pytest.mark.parametrize(
    "ptx_a, expected",
    [("f16", "p, 1, -1, 1;"), ("bf16", "p, 1, -1, 1;"), ("tf32", "p, 1, -1;"), ("e4m3", "p, 1, -1;"), ("s8", "p;"), ("u8", "p;")],
)
def test_dense_rs_tail(ptx_a, expected):
    from tilelang.contrib.cutedsl.gemm_v2 import _wgmma_tail

    assert _wgmma_tail(ptx_a, 1, -1, None, True) == expected


if __name__ == "__main__":
    tilelang.testing.main()
