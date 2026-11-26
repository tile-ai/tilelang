"""Frontend dispatch checks for tl.simd.vcvt that need no Ascend device.

The CCE ``::vcvt`` intrinsic covers the full documented type matrix (see
``src/tl_templates/ascend/simd_inst.h``, which dispatches straight to the
intrinsic), so every pair the docstring promises must dispatch on the Python
side. gh-3833: ``s32->f16`` and ``s64->f16`` were advertised but reached the
unsupported-conversion fallback.
"""

import pytest

import tilelang.ascend.language as T


def _build_convert(src_dtype: str, target_dtype: str):
    @T.prim_func
    def main():
        with T.Kernel(1), T.SimdVF():
            src = T.simd.alloc_var(src_dtype)
            dst = T.simd.vcvt(src, target_dtype)  # noqa: F841


@pytest.mark.parametrize(
    "src_dtype,target_dtype",
    [
        # the two pairs that raised before the fix (#3833)
        ("int32", "float16"),
        ("int64", "float16"),
        # already-working neighbours, kept as regression guards
        ("int16", "float16"),
        ("int32", "float32"),
        ("int64", "float32"),
    ],
)
def test_documented_int_to_float_pairs_dispatch(src_dtype, target_dtype):
    _build_convert(src_dtype, target_dtype)
