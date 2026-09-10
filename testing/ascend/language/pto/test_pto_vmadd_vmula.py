"""PTO VMI port of simdvf/test_simdvf_vmadd_vmula.py — FMA via T.vmi.

- ``vmula``: direct ``T.vmi.vmula(acc, a, b, mask)`` → acc + a*b
- ``vmadd``: composed ``vadd(vmul(acc, a), b)`` → acc*a + b
  (TileLang PTO VMI has no ``vmadd`` intrinsic)
"""

import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.ascend.language as T


N = 256


def fma_kernel(dtype, op: str):
    dtype_str = str(dtype)
    assert dtype_str in ("float32", "float16", "bfloat16")
    assert op in ("vmadd", "vmula")
    LANES = 64 if dtype_str == "float32" else 128

    @tilelang.jit(target="pto")
    def kernel(n: int):
        @T.prim_func
        def main(
            A: T.Tensor((n,), dtype),
            B: T.Tensor((n,), dtype),
            C: T.Tensor((n,), dtype),
        ):
            with T.Kernel(1):
                a_ub = T.alloc_shared((n,), dtype)
                b_ub = T.alloc_shared((n,), dtype)
                c_ub = T.alloc_shared((n,), dtype)

                T.copy(A, a_ub)
                T.copy(B, b_ub)
                T.copy(C, c_ub)

                with T.SimdVF():
                    mask = T.vmi.create_mask(LANES, size=LANES)
                    for i in T.serial(n // LANES):
                        c = T.vmi.vload(c_ub[i * LANES], size=LANES)
                        a = T.vmi.vload(a_ub[i * LANES], size=LANES)
                        b = T.vmi.vload(b_ub[i * LANES], size=LANES)
                        if op == "vmula":
                            out = T.vmi.vmula(c, a, b, mask)
                        else:
                            out = T.vmi.vadd(T.vmi.vmul(c, a, mask), b, mask)
                        T.vmi.vstore(out, c_ub[i * LANES], mask)

                T.copy(c_ub, C)

        return main

    return kernel


PARAMS = [
    ("vmadd", torch.float32),
    ("vmadd", torch.float16),
    ("vmadd", torch.bfloat16),
    ("vmula", torch.float32),
    ("vmula", torch.float16),
    ("vmula", torch.bfloat16),
]


def simulator_safe_randn(shape, *, dtype, device):
    return torch.randn(shape, dtype=dtype, device="cpu").to(device)


@pytest.mark.pto
@pytest.mark.parametrize("op,dtype", PARAMS)
def test_pto_fma(op, dtype):
    ref = (lambda a, b, c: c * a + b) if op == "vmadd" else (lambda a, b, c: c + a * b)
    device = torch.device("npu")
    torch.manual_seed(0)
    a = simulator_safe_randn(N, dtype=dtype, device=device)
    b = simulator_safe_randn(N, dtype=dtype, device=device)
    c_in = simulator_safe_randn(N, dtype=dtype, device=device)
    expected = ref(a.cpu(), b.cpu(), c_in.cpu())

    forward = fma_kernel(T.dtype(dtype), op)(N)
    forward(a, b, c_in)
    torch.npu.synchronize()

    if dtype == torch.float32:
        ok = torch.allclose(c_in.cpu(), expected, rtol=0, atol=1e-5)
    elif dtype == torch.float16:
        ok = torch.allclose(c_in.cpu(), expected, rtol=0, atol=1e-2)
    else:
        ok = torch.allclose(c_in.cpu(), expected, rtol=0, atol=1e-1)

    if not ok:
        diff = (c_in.cpu().float() - expected.float()).abs()
        print(f"\n{dtype}: max_diff={diff.max().item():.6f}")
    assert ok, f"{op} {dtype} failed"


if __name__ == "__main__":
    tilelang.testing.main()
