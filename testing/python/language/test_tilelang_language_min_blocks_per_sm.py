"""Tests for T.annotate_min_blocks_per_sm → __launch_bounds__(maxThreads, minBlocks)."""

import tilelang as tl
import tilelang.language as T
import tilelang.testing


@tl.jit(out_idx=[2], target="cuda")
def _kernel_min_blocks_per_sm():
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(128, threads=128) as bx:
            T.annotate_min_blocks_per_sm(2)
            for i in T.serial(128):
                C[bx, i] = A[bx, i] + B[bx, i]

    return main


@tilelang.testing.requires_cuda
def test_annotate_min_blocks_per_sm_launch_bounds():
    """Codegen should emit the second __launch_bounds__ argument from the annotation."""
    src = _kernel_min_blocks_per_sm.get_kernel_source()
    assert "__launch_bounds__(128, 2)" in src


if __name__ == "__main__":
    tilelang.testing.main()
