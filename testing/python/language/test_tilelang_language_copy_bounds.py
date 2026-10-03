import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


def _copy_program(src_start, dst_start):
    @T.prim_func
    def main(
        src: T.Tensor((8,), "int32"),
        dst: T.Tensor((8,), "int32"),
    ):
        with T.Kernel(1, threads=8):
            src_shared = T.alloc_shared((6,), "int32")
            T.copy(src[src_start : src_start + 6], src_shared)
            T.copy(src_shared, dst[dst_start : dst_start + 6])

    return main


def test_copy_rejects_out_of_bounds_source_region():
    with pytest.raises(ValueError, match="T.copy source region"):
        tilelang.compile(_copy_program(6, 0), target="cuda")


def test_copy_rejects_out_of_bounds_destination_region():
    with pytest.raises(ValueError, match="T.copy destination region"):
        tilelang.compile(_copy_program(0, 6), target="cuda")


@tilelang.testing.requires_cuda
def test_copy_accepts_in_bounds_regions():
    kernel = tilelang.compile(_copy_program(2, 2), target="cuda")
    assert kernel is not None
