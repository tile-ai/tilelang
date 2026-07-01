from pathlib import Path
import re

import torch
import tilelang.testing
import tilelang.language as T

def test_cuda_common_h_defines_bfloat16_rsqrt_overload():
    repo_root = Path(__file__).resolve().parents[3]
    common_h = repo_root / "src" / "tl_templates" / "cuda" / "common.h"
    source = common_h.read_text()

    pattern = re.compile(
        r"TL_PATCH\s+TL_DEVICE\s+bfloat16_t\s+hrsqrt\s*"
        r"\(\s*const\s+bfloat16_t\s+x\s*\)\s*\{"
    )
    assert pattern.search(source), "common.h must define hrsqrt(bfloat16_t) for bf16 T.rsqrt codegen"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(8, 0)
def test_bfloat16_rsqrt_compiles_and_runs():
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


def test_cuda_math_h_bfloat16_fast_exp_does_not_self_recurse():
    repo_root = Path(__file__).resolve().parents[3]
    common_h = repo_root / "src" / "tl_templates" / "cuda" / "common.h"
    math_h = repo_root / "src" / "tl_templates" / "cuda" / "math.h"
    common_source = common_h.read_text()
    source = math_h.read_text()

    # `hexp` is #define'd to cutlass::fast_exp, so the bf16 fast_exp overload
    # must route through float; calling `::hexp(x)` here recurses into itself.
    assert "#include <cutlass/fast_math.h>" not in common_source
    assert "#include <cutlass/fast_math.h>" in source
    assert "#define hexp cutlass::fast_exp" in source
    match = re.search(
        r"bfloat16_t\s+fast_exp\s*\(\s*bfloat16_t\s+x\s*\)\s*\{([^}]*)\}",
        source,
    )
    assert match, "math.h must define fast_exp(bfloat16_t) for bf16 T.exp codegen"
    body = match.group(1)
    assert "float(" in body, "bf16 fast_exp must route through float to avoid self-recursion"




@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(8, 0)
def test_bfloat16_exp_compiles_and_runs():
    n = 32

    @T.prim_func
    def main(A: T.Tensor((n,), T.bfloat16), B: T.Tensor((n,), T.bfloat16)):
        with T.Kernel(1, threads=32):
            values = T.alloc_fragment((n,), T.bfloat16)
            for i in T.Parallel(n):
                values[i] = T.exp(A[i])
            for i in T.Parallel(n):
                B[i] = values[i]

    kernel = tilelang.compile(main, out_idx=[1], target="cuda")
    inputs = torch.full((n,), 1.0, dtype=torch.bfloat16, device="cuda")
    actual = kernel(inputs)

    expected = torch.exp(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

if __name__ == "__main__":
    tilelang.testing.main()
