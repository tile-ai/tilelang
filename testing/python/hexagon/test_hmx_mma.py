"""Test for Hexagon HMX MMA compilation and lowering."""

import pytest
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.intrinsics.hexagon import hmx
from tilelang import tvm as tvm

# Shapes to test: (M, N, K)
# HMX v73 typically supports 32x32 tiles for int8 (32768 bits).
TEST_SHAPES = [
    (32, 32, 32),
    # (64, 64, 32),
    # to-do
]


def has_hexagon_codegen():
    """Check if LLVM and Hexagon are available in the current TVM build."""
    try:
        if not tvm.runtime.enabled("llvm"):
            return False
        # Try to construct a Hexagon target object
        tvm.target.Target("llvm -mtriple=hexagon -mcpu=hexagonv73")
        return True
    except Exception:
        return False


def build_hmx_matmul(M, N, K):
    """Kernel factory for HMX Matmul."""

    @T.prim_func
    def main(
        A_host: T.Tensor((M, K), "int8"),
        B_host: T.Tensor((K, N), "int8"),
        C_host: T.Tensor((M, N), "int32"),
    ):
        # Allocation in Hexagon Specific Scopes
        A_vtcm = T.alloc_fragment((M, K), "int8", scope="global.vtcm")
        B_vtcm = T.alloc_fragment((K, N), "int8", scope="global.vtcm")
        C_acc = T.alloc_fragment((M, N), "int32", scope="global.hmx.acc")

        # DDR -> VTCM
        for i, k in T.grid(M, K):
            A_vtcm[i, k] = A_host[i, k]
        for k, j in T.grid(K, N):
            B_vtcm[k, j] = B_host[k, j]

        # Init Accumulator
        for i, j in T.grid(M, N):
            C_acc[i, j] = T.cast(0, "int32")

        # Hardware MMA Intrinsic
        hmx.mma(A_vtcm, B_vtcm, C_acc)

        # HMX.acc -> DDR
        for i, j in T.grid(M, N):
            C_host[i, j] = C_acc[i, j]

    return main


@pytest.mark.skipif(not has_hexagon_codegen(), reason="Hexagon LLVM backend not available")
@pytest.mark.parametrize("M, N, K", TEST_SHAPES, ids=[f"shape={m}x{n}x{k}" for m, n, k in TEST_SHAPES])
def test_hmx_mma_compilation(M, N, K):
    """Verify that HMX MMA kernels lower to the correct Hexagon hardware intrinsics."""
    func = build_hmx_matmul(M, N, K)
    target = tvm.target.Target("llvm -mtriple=hexagon -mcpu=hexagonv73")

    # Compile the kernel
    kernel = tl.compile(func, target=target)

    # Validation Logic
    ir = kernel.kernel_source

    # Automated IR Checks
    assert 'target triple = "hexagon"' in ir, "Missing Hexagon target triple in IR"
    assert "A_vtcm" in ir, "VTCM allocation (A_vtcm) not found in IR"
    assert "C_acc" in ir, "Accumulator allocation (C_acc) not found in IR"
    assert "HexKL_mma_i8acc32" in ir, "Hexagon HMX hardware intrinsic call not found in IR"
    assert "hmx_mma_placeholder" not in ir, "HMX placeholder was not lowered"


@pytest.mark.skipif(not has_hexagon_codegen(), reason="Hexagon LLVM backend not available")
def test_hmx_host_execution_guard():
    """Verify that attempting to run Hexagon code on host raises a clear error."""
    func = build_hmx_matmul(32, 32, 32)
    target = tvm.target.Target("llvm -mtriple=hexagon -mcpu=hexagonv73")

    kernel = tl.compile(func, target=target)

    # This should raise the RuntimeError from base.py
    with pytest.raises(RuntimeError, match="Hexagon kernels cannot be executed directly on the host"):
        kernel(None, None, None)


if __name__ == "__main__":
    tilelang.testing.main()
