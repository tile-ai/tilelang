"""Minimal reproducer for MFMA DataType vs str bug.

Bug: MatrixCoreIntrinEmitter.mfma() passes self.a_dtype (a tvm_ffi DataType
object, which is a str subclass) to T.tvm_mfma(). The TVM FFI C++ layer
dispatches DataType before str in its type conversion, so it never gets
auto-converted to StringImm (a PrimExpr). This causes:

    TypeError: Mismatched type on argument #2 when calling:
    `tirx.Call(...)`. Expected `Array<ir.PrimExpr>` but got
    `Array[index 3: DataType]`

The bug is triggered whenever local_size_a/b/out == 1, because then
compute_a_dtype = self.a_dtype (DataType) instead of f"{self.a_dtype}x{N}"
(str). This happens with certain GEMM shapes where the MFMA tile exactly
matches the data width.

Fix: convert self.a_dtype/b_dtype/accum_dtype to str() before building
the compute dtype strings.

To run:
    python testing/python/amd/test_mfma_dtype_str_bug.py
    # or
    pytest testing/python/amd/test_mfma_dtype_str_bug.py -v
"""

import pytest
import torch
import tilelang
import tilelang.language as T


def matmul_kernel(M, N, K, in_dtype, out_dtype):
    """Minimal GEMM kernel that triggers the MFMA dtype bug."""

    @T.prim_func
    def kernel(
        A: T.Tensor[(M, K), in_dtype],
        B: T.Tensor[(K, N), in_dtype],
        C: T.Tensor[(M, N), out_dtype],
    ):
        with T.Kernel(1, threads=64) as _:
            A_shared = T.alloc_shared((M, K), in_dtype)
            B_shared = T.alloc_shared((K, N), in_dtype)
            C_local = T.alloc_fragment((M, N), out_dtype)

            T.clear(C_local)
            T.copy(A[0, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[0, 0])

    return kernel


@pytest.mark.skipif(
    torch.version.hip is None,
    reason="MFMA is AMD ROCm only",
)
@pytest.mark.parametrize(
    "in_dtype,out_dtype",
    [
        (T.float32, T.float32),
        (T.float16, T.float32),
        (T.bfloat16, T.float32),
    ],
    ids=["f32", "f16", "bf16"],
)
def test_mfma_compile(in_dtype, out_dtype):
    """Verify that a simple GEMM using MFMA compiles on ROCm.

    Without the fix, this fails with:
        TypeError: Expected Array<ir.PrimExpr> but got Array[index 3: DataType]
    """
    M, N, K = 16, 16, 16
    func = matmul_kernel(M, N, K, in_dtype, out_dtype)
    mod = tilelang.compile(func)
    assert mod is not None


@pytest.mark.skipif(
    torch.version.hip is None,
    reason="MFMA is AMD ROCm only",
)
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required",
)
@pytest.mark.parametrize(
    "in_dtype,out_dtype,torch_in,torch_out",
    [
        (T.float32, T.float32, torch.float32, torch.float32),
        (T.float16, T.float32, torch.float16, torch.float32),
        (T.bfloat16, T.float32, torch.bfloat16, torch.float32),
    ],
    ids=["f32", "f16", "bf16"],
)
def test_mfma_correctness(in_dtype, out_dtype, torch_in, torch_out):
    """Verify MFMA GEMM produces correct results."""
    M, N, K = 16, 16, 16
    func = matmul_kernel(M, N, K, in_dtype, out_dtype)
    mod = tilelang.compile(func)

    A = torch.randn(M, K, dtype=torch_in, device="cuda")
    B = torch.randn(K, N, dtype=torch_in, device="cuda")
    C = torch.empty(M, N, dtype=torch_out, device="cuda")

    mod(A, B, C)
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).to(torch_out)
    torch.testing.assert_close(C, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    if torch.version.hip is None:
        print("SKIP: not a ROCm environment")
        exit(0)

    print("=== Reproduce: compile MFMA GEMM kernels ===")
    failed = False
    for name, in_dt, out_dt in [
        ("f32", T.float32, T.float32),
        ("f16", T.float16, T.float32),
        ("bf16", T.bfloat16, T.float32),
    ]:
        try:
            func = matmul_kernel(16, 16, 16, in_dt, out_dt)
            mod = tilelang.compile(func)
            print(f"  {name}: COMPILE OK")
        except TypeError as e:
            if "DataType" in str(e):
                print(f"  {name}: BUG REPRODUCED - {e}")
                failed = True
            else:
                raise

    if torch.cuda.is_available():
        print("\n=== Correctness ===")
        for name, in_dt, out_dt, t_in, t_out in [
            ("f32", T.float32, T.float32, torch.float32, torch.float32),
            ("f16", T.float16, T.float32, torch.float16, torch.float32),
            ("bf16", T.bfloat16, T.float32, torch.bfloat16, torch.float32),
        ]:
            func = matmul_kernel(16, 16, 16, in_dt, out_dt)
            mod = tilelang.compile(func)
            A = torch.randn(16, 16, dtype=t_in, device="cuda")
            B = torch.randn(16, 16, dtype=t_in, device="cuda")
            C = torch.empty(16, 16, dtype=t_out, device="cuda")
            mod(A, B, C)
            torch.cuda.synchronize()
            ref = (A.float() @ B.float()).to(t_out)
            diff = (C - ref).abs().max().item()
            status = "PASS" if diff < 0.1 else "FAIL"
            print(f"  {name}: max_diff={diff:.6f} {status}")
            if status == "FAIL":
                failed = True

    if failed:
        exit(1)
