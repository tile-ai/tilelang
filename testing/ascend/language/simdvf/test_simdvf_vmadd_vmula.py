"""pytest test for T.simd.vmadd and T.simd.vmula — SIMD vector FMA on dav-3510."""

import pytest
import torch
import tilelang
import tilelang.ascend.language as T


N = 8192


def fma_kernel(dtype, op: str):
    """Unified kernel for vmadd (C = C*A + B) and vmula (C = C + A*B)."""
    dtype_str = str(dtype)
    assert dtype_str in ("float32", "float16", "bfloat16")
    assert op in ("vmadd", "vmula")

    if dtype_str == "float32":
        pset_width = 32
        ld_dist = "NORM"
        st_dist = "NORM_B32"
    else:
        pset_width = 16
        ld_dist = "NORM"
        st_dist = "NORM_B16"

    @tilelang.jit
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
                    mask = T.simd.pset(pset_width)
                    for i in T.serial(n // 64):
                        c = T.simd.alloc_var(dtype)
                        c = T.simd.vld(c_ub[i * 64], dist=ld_dist)
                        a = T.simd.vld(a_ub[i * 64], dist=ld_dist)
                        b = T.simd.vld(b_ub[i * 64], dist=ld_dist)
                        getattr(T.simd, op)(c, a, b, mask)
                        T.simd.vsts(c_ub[i * 64], c, mask, dist=st_dist)

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


@pytest.mark.parametrize("op,dtype", PARAMS)
def test_simdvf_fma(op, dtype):
    ref = (lambda a, b, c: c * a + b) if op == "vmadd" else (lambda a, b, c: c + a * b)
    device = torch.device("npu")
    torch.manual_seed(0)
    a = torch.randn(N, dtype=dtype, device=device)
    b = torch.randn(N, dtype=dtype, device=device)
    c_in = torch.randn(N, dtype=dtype, device=device)
    expected = ref(a, b, c_in)

    forward = fma_kernel(T.dtype(dtype), op)(N)
    forward(a, b, c_in)
    print(forward.get_kernel_source())
    torch.npu.synchronize()

    if dtype == torch.float32:
        ok = torch.allclose(c_in, expected, rtol=0, atol=1e-5)
    elif dtype == torch.float16:
        ok = torch.allclose(c_in, expected, rtol=0, atol=1e-2)
    else:
        ok = torch.allclose(c_in, expected, rtol=0, atol=1e-1)

    if not ok:
        diff = (c_in.float() - expected.float()).abs()
        print(f"\n{dtype}: max_diff={diff.max().item():.6f}")
        print(f"  first 5 c_in: {c_in[:5].float().tolist()}")
        print(f"  first 5 ref : {expected[:5].float().tolist()}")
    assert ok, f"{op} {dtype} failed"


if __name__ == "__main__":
    for op, dtype in PARAMS:
        test_simdvf_fma(op, dtype)
    print("PASS: test_simdvf_fma")
