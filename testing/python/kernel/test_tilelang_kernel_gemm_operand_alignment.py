"""A shared ``T.gemm`` operand region must start on a 16-byte boundary.

ldmatrix needs 16-byte aligned row addresses and the WGMMA descriptor stores the
start address as ``addr >> 4``. A K offset of 4 fp16 elements (8 bytes) used to
fault with ``misaligned address`` on the MMA path and was silently truncated on
the WGMMA path, returning the product of the wrong slice. It is now rejected at
lowering, using the physical origin under the selected shared layout.
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing

M, N, K = 64, 32, 32
KS = K // 2

_MISALIGNED = r"T\.gemm\(\) operand {operand} region .* which is never a multiple of 16"


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


def _symbolic_offset_gemm():
    """The byte offset is 4 or 8 depending on ``selector``; neither is aligned."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((KS, N), T.float16),
        C: T.Tensor((M, N), T.float32),
        selector: T.int32,
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((M, K), T.float16)
            B_shared = T.alloc_shared((KS, N), T.float16)
            C_local = T.alloc_fragment((M, N), T.float32)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            off = (selector % 2) * 2 + 2
            T.gemm(A_shared[:, off : off + KS], B_shared, C_local, clear_accum=True)
            T.copy(C_local, C)

    return main


def _padded_layout_gemm():
    """Logical row pitch 36 (72 B) but physical pitch 40 (80 B): row 1 is aligned."""

    @T.prim_func
    def main(
        A: T.Tensor((65, 36), T.float16),
        B: T.Tensor((32, 32), T.float16),
        C: T.Tensor((64, 32), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((65, 36), T.float16)
            B_shared = T.alloc_shared((32, 32), T.float16)
            C_local = T.alloc_fragment((64, 32), T.float32)
            T.annotate_layout({A_shared: T.Layout((65, 36), lambda i, j: i * 40 + j)})
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared[1:65, :32], B_shared, C_local, clear_accum=True)
            T.copy(C_local, C)

    return main


def _compile(func, arch):
    return tilelang.compile(func, target={"kind": "cuda", "arch": arch})


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_80", "sm_90"])
@pytest.mark.parametrize("operand, a_off, b_off", [("A", 4, 0), ("B", 0, 4)])
def test_gemm_rejects_misaligned_operand_origin(arch, operand, a_off, b_off):
    with pytest.raises(ValueError, match=_MISALIGNED.format(operand=operand)):
        _compile(_sliced_gemm(a_off, b_off), arch)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_80", "sm_90"])
def test_gemm_rejects_provably_misaligned_symbolic_origin(arch):
    with pytest.raises(ValueError, match=_MISALIGNED.format(operand="A")):
        _compile(_symbolic_offset_gemm(), arch)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("arch", ["sm_80", "sm_90"])
@pytest.mark.parametrize("a_off, b_off", [(8, 0), (0, 8)])
def test_gemm_accepts_aligned_operand_origin(arch, a_off, b_off):
    _compile(_sliced_gemm(a_off, b_off), arch)


@tilelang.testing.requires_cuda
def test_gemm_accepts_padded_layout_with_aligned_physical_origin():
    """The check must use the annotated layout's pitch, not the logical shape."""
    _compile(_padded_layout_gemm(), "sm_80")


if __name__ == "__main__":
    tilelang.testing.main()
