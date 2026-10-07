"""``T.copy`` must not read or write outside its src and dst regions.

The copy loop takes its extents from one region and indexes the other at the
same offsets. When the loop region is larger than the other one, the copy used
to compile silently and read outside src or write outside dst (#3301, F027).
A loop region that is smaller only copies part of the other region; that stays
accepted (#1883).
"""

import pytest

import tilelang
import tilelang.language as T
import tilelang.testing

_OUTSIDE = (
    r"\[TileLang Semantic Check\] T\.copy would {action} outside the {side} region .* "
    r"non-unit dimension {dim} has extent {loop}, but the {side} region has extent {other}\."
)


def _compile(func):
    return tilelang.compile(func, target={"kind": "cuda", "arch": "sm_80"})


def _global_to_shared_over_read():
    @T.prim_func
    def main(A: T.Tensor((128, 128), T.float32), B: T.Tensor((64, 64), T.float32)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float32)
            T.copy(A[0:32, 0:64], S)
            T.copy(S, B)

    return main


def _global_to_larger_shared():
    @T.prim_func
    def main(A: T.Tensor((64, 64), T.float32), B: T.Tensor((128, 128), T.float32)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((128, 128), T.float32)
            T.copy(A[0:64, 0:64], S)
            T.copy(S, B)

    return main


def _shared_to_smaller_global_region():
    @T.prim_func
    def main(A: T.Tensor((64, 64), T.float32), B: T.Tensor((128, 128), T.float32)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float32)
            T.copy(A, S)
            T.copy(S, B[0:32, 0:32])

    return main


def _global_to_global_1d():
    @T.prim_func
    def main(A: T.Tensor((256,), T.float32), B: T.Tensor((256,), T.float32)):
        with T.Kernel(1, threads=128):
            T.copy(A[0:55], B[0:16])

    return main


def _async_copy_over_read():
    @T.prim_func
    def main(A: T.Tensor((128, 128), T.float32), B: T.Tensor((64, 64), T.float32)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float32)
            T.async_copy(A[0:32, 0:64], S)
            T.copy(S, B)

    return main


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "make, action, side, loop, other",
    [
        (_global_to_shared_over_read, "read", "src", 64, 32),
        (_global_to_larger_shared, "read", "src", 128, 64),
        (_shared_to_smaller_global_region, "write", "dst", 64, 32),
        (_global_to_global_1d, "write", "dst", 55, 16),
        (_async_copy_over_read, "read", "src", 64, 32),
    ],
)
def test_copy_rejects_access_outside_region(make, action, side, loop, other):
    pattern = _OUTSIDE.format(action=action, side=side, dim=0, loop=loop, other=other)
    with pytest.raises(ValueError, match=pattern):
        _compile(make())


@tilelang.testing.requires_cuda
def test_copy_accepts_partial_write_of_larger_dst():
    """A smaller loop region writes only part of dst; #1883 keeps this legal."""

    @T.prim_func
    def main(A: T.Tensor((64, 64), T.float32), B: T.Tensor((128, 128), T.float32)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float32)
            T.copy(A, S)
            T.copy(S, B[0:128, 0:128])

    _compile(main)


@tilelang.testing.requires_cuda
def test_copy_accepts_unit_dims_and_inferred_extents():
    """Unit dims are skipped, and a start-only source takes its extents from dst."""

    @T.prim_func
    def main(A: T.Tensor((4, 64, 64), T.float32), B: T.Tensor((64, 64), T.float32)):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((64, 64), T.float32)
            R = T.alloc_shared((64,), T.float32)
            T.copy(A[1, 0:64, 0:64], S)
            T.copy(A[2, 0, 0], S)
            T.copy(A[3, 5, 0:64], R)
            T.copy(S, B)

    _compile(main)


@tilelang.testing.requires_cuda
def test_copy_accepts_extent_not_provably_larger():
    """A symbolic loop extent that cannot be proven larger is left alone."""

    @T.prim_func
    def main(A: T.Tensor((256,), T.float32), B: T.Tensor((256,), T.float32), n: T.int32):
        with T.Kernel(1, threads=128):
            T.copy(A[0:n], B[0:16])

    _compile(main)


if __name__ == "__main__":
    tilelang.testing.main()
