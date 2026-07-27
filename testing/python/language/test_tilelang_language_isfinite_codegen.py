import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _get_isfinite_source() -> str:
    n = 1

    @T.prim_func
    def main(
        x: T.Tensor((n,), T.float32),
        y: T.Tensor((n,), T.int32),
    ):
        with T.Kernel(1, threads=1):
            pred = T.alloc_var(T.bool)
            pred = T.isfinite(x[0])
            y[0] = T.if_then_else(pred, 1, 0)

    artifact = tilelang.lower(main, target="cuda")
    return artifact.kernel_source


def _get_isfinite_expr(code: str) -> str:
    pattern = r"\b\w+\s*=\s*(.*\bisfinite\s*\(.*\));"
    for line in code.splitlines():
        match = re.search(pattern, line)
        if match:
            return match.group(1)
    raise AssertionError("Failed to find CUDA isfinite call in generated source")


@tilelang.testing.requires_cuda
def test_isfinite_codegen_uses_cuda_intrinsic():
    """Check T.isfinite lowers to CUDA's isfinite for float32."""
    src = _get_isfinite_source()
    expr = _get_isfinite_expr(src)

    print("=== isfinite codegen ===")
    print(src)
    print("=== extracted expression ===")
    print(expr)

    assert "isfinite(" in expr
    assert "fabsf(" not in expr
    assert "CUDART_INF_F" not in expr
    assert "!= x[0]" not in expr
    assert "x[0] != x[0]" not in expr


if __name__ == "__main__":
    tilelang.testing.main()
