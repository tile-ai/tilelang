from pathlib import Path
import re

import pytest
import torch


def test_cuda_common_h_defines_bfloat16_rsqrt_overload():
    repo_root = Path(__file__).resolve().parents[3]
    common_h = repo_root / "src" / "tl_templates" / "cuda" / "common.h"
    source = common_h.read_text()

    pattern = re.compile(
        r"TL_PATCH\s+TL_DEVICE\s+bfloat16_t\s+hrsqrt\s*"
        r"\(\s*const\s+bfloat16_t\s+x\s*\)\s*\{"
    )
    assert pattern.search(source), "common.h must define hrsqrt(bfloat16_t) for bf16 T.rsqrt codegen"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_bfloat16_rsqrt_compiles_and_runs():
    import tilelang
    import tilelang.language as T

    if torch.cuda.get_device_capability() < (8, 0):
        pytest.skip("bfloat16 CUDA math requires compute capability >= 8.0")

    n = 32

    @T.prim_func
    def main(A: T.Tensor((n,), T.bfloat16), B: T.Tensor((n,), T.bfloat16)):
        with T.Kernel(1, threads=32):
            values = T.alloc_fragment((n,), T.bfloat16)
            for i in T.Parallel(n):
                values[i] = T.rsqrt(A[i])
            for i in T.Parallel(n):
                B[i] = values[i]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    inputs = torch.full((n,), 4.0, dtype=torch.bfloat16, device="cuda")
    actual = kernel(inputs)

    torch.testing.assert_close(actual, torch.full_like(actual, 0.5))
