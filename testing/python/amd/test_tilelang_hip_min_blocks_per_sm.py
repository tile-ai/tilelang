"""T.annotate_min_blocks_per_sm on HIP: blocks per compute unit to waves per EU."""

import pytest
import tilelang as tl
import tilelang.rocm.language as T
import tilelang.testing
from tilelang.backend.target import determine_target
from tilelang.rocm.target import target_get_warp_size


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


def _wave_size():
    # The wave size the codegen uses, not an arch-name prefix: gfx1250/gfx1251
    # (CDNA5) share the gfx12 prefix with RDNA.
    return target_get_warp_size(determine_target("auto", return_object=True))


# Expected waves per EU by wave size. Both divide by 4 SIMDs: a CDNA compute unit
# (wave64) or an RDNA workgroup processor (wave32, the default WGP mode).
_WAVE64_CASES = [
    (64, 4, 1),
    (128, 2, 1),
    (128, 4, 2),
    (256, 1, 1),
    (256, 2, 2),
    (256, 3, 3),
    (512, 1, 2),
    (512, 2, 4),
    (768, 1, 3),
]
_WAVE32_CASES = [
    (64, 2, 1),
    (64, 4, 2),
    (128, 1, 1),
    (128, 2, 2),
    (256, 1, 2),
    (512, 1, 4),
]


@tilelang.testing.requires_rocm
def test_annotation_maps_blocks_per_cu_to_waves_per_eu():
    for threads, blocks, waves_per_eu in _WAVE64_CASES if _wave_size() == 64 else _WAVE32_CASES:
        src = _annotated(threads, blocks).get_kernel_source()
        assert f"__launch_bounds__({threads}, {waves_per_eu})" in src, (threads, blocks)


@tilelang.testing.requires_rocm
def test_no_annotation_keeps_single_argument():
    src = _plain(256).get_kernel_source()
    assert "__launch_bounds__(256)" in src


if __name__ == "__main__":
    tilelang.testing.main()
