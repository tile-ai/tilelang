"""Two NPU tests that compare on-path FixPipe quant vs SimdVF cast.

Fused Linear epilogue ``Y = X @ W^T + bias`` in float16 or bfloat16:

1. Two consecutive non-dual FixPipes + on-path quant + SimdVF vadd.
2. One hardware dual_copy (fp32) + SimdVF vcvt + the same SimdVF vadd.

Both kernels share the GEMM scaffolding; only the L0C->UB cast path differs.
"""

import pytest

from example_dual_copy_onpath_vs_vf_cast import run_one

M, K, N = 8192, 256, 8192
DTYPE = "bfloat16"
DST_CASES = (("float16", 2e-2), ("bfloat16", 5e-2))


@pytest.mark.parametrize("dst_dtype,thresh", DST_CASES)
def test_onpath_two_fixpipe_quant_plus_vf(dst_dtype, thresh):
    """L0C fp32 -> two FixPipes with quant_pre -> SimdVF vadd -> GM."""
    print(f"\n=== test 1: two FixPipes + on-path quant + SimdVF vadd (dst={dst_dtype}) ===")
    run_one(M, K, N, DTYPE, dst_dtype, "onpath", thresh=thresh)


@pytest.mark.parametrize("dst_dtype,thresh", DST_CASES)
def test_hardware_dual_vf_cast_plus_vf(dst_dtype, thresh):
    """L0C fp32 -> one hardware dual -> SimdVF vcvt -> SimdVF vadd -> GM."""
    print(f"\n=== test 2: one hardware dual + SimdVF vcvt + SimdVF vadd (dst={dst_dtype}) ===")
    run_one(M, K, N, DTYPE, dst_dtype, "vf_cast", thresh=thresh)


if __name__ == "__main__":
    from example_dual_copy_onpath_vs_vf_cast import compare_epilogues

    for dst_dtype, _ in DST_CASES:
        compare_epilogues(M, K, N, dtype=DTYPE, dst_dtype=dst_dtype)
    print("PASS: onpath vs vf_cast")
