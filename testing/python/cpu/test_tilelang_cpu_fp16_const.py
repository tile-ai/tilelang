"""CPU codegen float literal regression tests.

Covers two paths through CodeGenTileLangC::VisitExpr_(FloatImmNode*) that have
broken before:

1. Finite fp16 constants must lower to the cast form `(half)1.5e+00f`
   (legal C/C++), NOT the `1.5e+00h` form copied from Metal — the `h`
   suffix is MSL-only and g++ rejects it
   ("no matching literal operator ... operator\"\"h").
2. +/-inf and nan must lower to the C99 macros INFINITY / NAN, NOT the
   base class's "-inff" / "nanf" output (IEEE text + 'f' suffix), which
   g++ rejects ("use of undeclared identifier 'inff'"). This path is
   exercised by reduce max/min identity init values.

Both were latent because no prior c-target test emitted a finite fp16
FloatImm or an inf identity. reduce max/min introduced the inf path; the
fp16 finite-constant regression was introduced by copying Metal's FloatImm
override verbatim and fixed by delegating finite values back to the base
class.
"""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang import tvm


def _compile_c(func, out_idx):
    with tvm.target.Target("c"):
        return tilelang.compile(
            func,
            out_idx=out_idx,
            target="c",
            target_host="c",
            execution_backend="cython",
        )


def _source(func) -> str:
    with tvm.target.Target("c"):
        artifact = tilelang.lower(func)
    return artifact.kernel_source


def test_fp16_finite_constant_arithmetic():
    """Reproducer: T.float16(1.5) + T.float16(2.25) must compile and run.

    Regression for the `1.5e+00h` literal bug: the Metal-derived override
    appended an 'h' suffix for fp16, which is invalid in C/C++.
    """

    @T.prim_func
    def main(A: T.Tensor((4,), "float16"), B: T.Tensor((4,), "float16")):
        with T.Kernel(1):
            for i in T.grid(4):
                B[i] = A[i] + T.float16(1.5) + T.float16(2.25)

    # Must compile cleanly (the bug failed at g++ compile step).
    kernel = _compile_c(main, out_idx=[1])

    # The generated source must use the legal cast form, not the 'h' suffix.
    src = _source(main)
    assert "(half)" in src, f"expected (half) cast form in source:\n{src}"
    assert not re.search(r"[0-9]h\b", src), f"source must not contain an 'h' literal suffix (MSL-only):\n{src}"

    a = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float16)
    out = kernel(a)
    expected = a + torch.tensor(3.75, dtype=torch.float16)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("op", ["max", "min"])
def test_fp16_reduce_inf_identity(op):
    """fp16 reduce_max/min identity (+/-inf) must lower to INFINITY, not inff.

    The base class prints "-inf" + 'f' => "-inff", which g++ rejects.
    """

    @T.prim_func
    def main(
        A: T.Tensor((4, 8), "float16"),
        B: T.Tensor((4,), "float16"),
    ):
        with T.Kernel(1):
            src = T.alloc_local((4, 8), "float16")
            dst = T.alloc_local((4,), "float16")
            for i, j in T.grid(4, 8):
                src[i, j] = A[i, j]
            if op == "max":
                T.reduce_max(src, dst, dim=1, clear=True)
            else:
                T.reduce_min(src, dst, dim=1, clear=True)
            for i in T.grid(4):
                B[i] = dst[i]

    # Source must use the C99 macro form, not the broken literal.
    src = _source(main)
    assert "INFINITY" in src, f"expected INFINITY macro for {op} identity in source:\n{src}"
    assert "inff" not in src, f"source must not contain 'inff':\n{src}"

    kernel = _compile_c(main, out_idx=[1])
    a = torch.randn((4, 8), dtype=torch.float16)
    out = kernel(a)
    expected = a.amax(dim=1) if op == "max" else a.amin(dim=1)
    torch.testing.assert_close(out, expected)


def test_float32_finite_constant_sanity():
    """float32 finite constants must keep the 'f' suffix after delegation.

    Guards against over-narrowing the override: delegating finite values to
    the base class must still produce `1.5e+00f` for float32. An input
    tensor is threaded through so the torch wrapper can infer the CPU
    device (a no-arg call defaults to CUDA).
    """

    @T.prim_func
    def main(A: T.Tensor((1,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            B[0] = A[0] + T.cast(1.5, "float32") + T.cast(2.25, "float32")

    src = _source(main)
    assert re.search(r"[0-9]f\b", src), f"expected a float32 'f' suffix literal in source:\n{src}"

    kernel = _compile_c(main, out_idx=[1])
    a = torch.tensor([0.0], dtype=torch.float32)
    out = kernel(a)
    torch.testing.assert_close(out, torch.tensor([3.75], dtype=torch.float32))


if __name__ == "__main__":
    tilelang.testing.main()
