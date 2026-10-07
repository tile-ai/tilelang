"""A shared ``T.gemm`` operand region must start on a 16-byte boundary.

ldmatrix needs 16-byte aligned row addresses and the WGMMA descriptor stores the
start address as ``addr >> 4``. A K offset of 4 fp16 elements (8 bytes) used to
fault with ``misaligned address`` on the MMA path and was silently truncated on
the WGMMA path, returning the product of the wrong slice. It is now rejected at
lowering.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing
import tvm

M, N, K = 64, 32, 32
KS = K // 2


def _sliced_gemm(a_off, b_off):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((KS, 2 * N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), T.float16)
            B_shared = T.alloc_shared((KS, 2 * N), T.float16)
            C_local = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(A_shared[:, a_off : a_off + KS], B_shared[:, b_off : b_off + N], C_local)
            T.copy(C_local, C)

    return main


def _compile(a_off, b_off, arch):
    return tilelang.compile(_sliced_gemm(a_off, b_off), target={"kind": "cuda", "arch": arch})


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_80", "sm_90"])
@pytest.mark.parametrize("operand, a_off, b_off", [("A", 4, 0), ("B", 0, 4)])
def test_gemm_rejects_misaligned_operand_origin(arch, operand, a_off, b_off):
    with pytest.raises(tvm.error.InternalError, match=rf"T\.gemm\(\) operand {operand} region .* starts 8 bytes past a 16-byte boundary"):
        _compile(a_off, b_off, arch)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_80", "sm_90"])
@pytest.mark.parametrize("a_off, b_off", [(8, 0), (0, 8)])
def test_gemm_accepts_aligned_operand_origin(arch, a_off, b_off):
    _compile(a_off, b_off, arch)


if __name__ == "__main__":
    tilelang.testing.main()
