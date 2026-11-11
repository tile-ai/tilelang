"""T.annotate_min_blocks_per_sm on HIP: blocks per compute unit to waves per EU."""

import pytest
import tilelang as tl
import tilelang.rocm.language as T
import tilelang.testing


@tl.jit(out_idx=[-1], target="hip")
def _annotated(threads, blocks):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(128, threads=threads) as bx:
            T.annotate_min_blocks_per_sm(blocks)
            for i in T.serial(128):
                C[bx, i] = A[bx, i] + B[bx, i]

    return main


@tl.jit(out_idx=[-1], target="hip")
def _plain(threads):
    @T.prim_func
    def main(
        A: T.Tensor((128, 128), "float32"),
        B: T.Tensor((128, 128), "float32"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(128, threads=threads) as bx:
            for i in T.serial(128):
                C[bx, i] = A[bx, i] + B[bx, i]

    return main


@tilelang.testing.requires_rocm
@pytest.mark.parametrize(
    ("threads", "blocks", "waves_per_eu"),
    [
        (64, 4, 1),
        (128, 2, 1),
        (128, 4, 2),
        (256, 1, 1),
        (256, 2, 2),
        (256, 3, 3),
        (512, 1, 2),
        (512, 2, 4),
        (768, 1, 3),
    ],
)
def test_annotation_maps_blocks_per_cu_to_waves_per_eu(threads, blocks, waves_per_eu):
    src = _annotated(threads, blocks).get_kernel_source()
    assert f"__launch_bounds__({threads}, {waves_per_eu})" in src


@tilelang.testing.requires_rocm
def test_no_annotation_keeps_single_argument():
    src = _plain(256).get_kernel_source()
    assert "__launch_bounds__(256)" in src


if __name__ == "__main__":
    tilelang.testing.main()
