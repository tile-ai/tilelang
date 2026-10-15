import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


def _gemm_kernel(threads):

    @T.prim_func
    def main(
        A: T.Tensor((16, 32), "float16"),
        B: T.Tensor((32, 32), "float16"),
        C: T.Tensor((16, 32), "float32"),
    ):
        with T.Kernel(1, threads=threads):
            A_shared = T.alloc_shared((16, 32), "float16")
            B_shared = T.alloc_shared((32, 32), "float16")
            C_local = T.alloc_fragment((16, 32), "float32")
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
            T.copy(C_local, C)

    return main


@tilelang.testing.requires_cdna
def test_sub_wavefront_block_size_raises_instead_of_crashing():
    """A block narrower than one wavefront used to abort the process with SIGFPE.

    `block_size / 64` floors to zero warps on CDNA, and the resulting
    `M % (m_warp * kMPerWarp)` divided by zero. It must be a normal error.

    CDNA-only: 32 threads is a full wavefront on RDNA, so the guard correctly
    does not fire there.
    """
    with pytest.raises(Exception, match="wavefront"):
        tilelang.compile(_gemm_kernel(32), target="hip")


@tilelang.testing.requires_rocm
def test_full_wavefront_block_size_still_compiles():
    assert tilelang.compile(_gemm_kernel(64), target="hip") is not None


if __name__ == "__main__":
    tilelang.testing.main()
