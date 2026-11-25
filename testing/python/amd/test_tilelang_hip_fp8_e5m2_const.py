"""Regression tests for float8_e5m2 / float8_e5m2fnuz constants in HIP codegen."""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm
from tilelang.language.fp8 import determine_fp8_type


def _clear(dtype):
    @T.prim_func
    def main(O: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=128):
            x = T.alloc_fragment((128,), dtype)
            T.clear(x)
            T.copy(x, O)

    return main


def _fill(dtype):
    @T.prim_func
    def main(O: T.Tensor((2048,), dtype)):
        with T.Kernel(1, threads=128):
            x = T.alloc_fragment((2048,), dtype)
            T.fill(x, 1.5)
            T.copy(x, O)

    return main


def _masked_copy(dtype):
    @T.prim_func
    def main(X: T.Tensor((100, 64), dtype), O: T.Tensor((128, 64), dtype)):
        with T.Kernel(2, threads=128) as bx:
            s = T.alloc_shared((64, 64), dtype)
            T.copy(X[bx * 64, 0], s)
            T.copy(s, O[bx * 64, 0])

    return main


KERNELS = {"clear": _clear, "fill": _fill, "masked_copy": _masked_copy}


@pytest.mark.parametrize("mcpu, dtype", [("gfx942", "float8_e5m2fnuz"), ("gfx950", "float8_e5m2")])
@pytest.mark.parametrize("kernel", list(KERNELS))
def test_fp8_e5m2_const_lower(mcpu, dtype, kernel):
    """E5M2 constants are printed through fp8_e5_t (no GPU needed)."""
    if tvm.get_global_func("target.build.tilelang_hip_without_compile", allow_missing=True) is None:
        pytest.skip("TileLang HIP codegen is not enabled")
    target = {"kind": "hip", "mcpu": mcpu}
    with tvm.transform.PassContext(opt_level=3), tvm.target.Target(target):
        src = tilelang.lower(KERNELS[kernel](dtype), target=target).kernel_source
    assert "fp8_e5_t(" in src


@tilelang.testing.requires_cdna
def test_fp8_e5m2_const_correctness():
    dtype = determine_fp8_type("e5m2")
    torch_dtype = getattr(torch, str(dtype))

    O = torch.full((128,), 7.0, device="cuda").to(torch_dtype)
    tilelang.compile(_clear(dtype))(O)
    assert (O.float() == 0).all()

    O = torch.zeros(2048, device="cuda").to(torch_dtype)
    tilelang.compile(_fill(dtype))(O)
    assert (O.float() == 1.5).all()

    X = torch.randn(100, 64, device="cuda").to(torch_dtype)
    O = torch.full((128, 64), 7.0, device="cuda").to(torch_dtype)
    tilelang.compile(_masked_copy(dtype))(X, O)
    assert torch.equal(O[:100].float(), X.float())
    assert (O[100:].float() == 0).all()


if __name__ == "__main__":
    tilelang.testing.main()
