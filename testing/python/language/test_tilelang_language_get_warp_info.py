from typing import Optional

import tilelang.language as T
import tilelang.testing
import torch
from tilelang.utils.target import check_hip_availability

_IS_HIP_AVAILABLE = check_hip_availability()
_DEFAULT_WARPS_PER_GROUP = 4


def _resolve_warp_size(warp_size: Optional[int]) -> int:
    if warp_size is not None:
        return int(warp_size)
    return 64 if _IS_HIP_AVAILABLE else 32


def _resolve_warps_per_group(warps_per_group: Optional[int]) -> int:
    if warps_per_group is not None:
        return int(warps_per_group)
    return _DEFAULT_WARPS_PER_GROUP


@tilelang.jit(out_idx=[-1])
def _get_laneid_kernel(num_threads: int = 128, warp_size: Optional[int] = None):
    @T.prim_func
    def laneid_kernel(A: T.Tensor((num_threads,), T.int32)):
        with T.Kernel(1, threads=num_threads) as _:
            tx = T.get_thread_binding()
            A[tx] = T.get_lane_idx(warp_size)

    return laneid_kernel


@tilelang.jit(out_idx=[-1])
def _get_warp_idx_sync_kernel(num_threads: int = 128, warp_size: Optional[int] = None):
    @T.prim_func
    def warp_idx_sync_kernel(A: T.Tensor((num_threads,), T.int32)):
        with T.Kernel(1, threads=num_threads) as _:
            tx = T.get_thread_binding()
            A[tx] = T.get_warp_idx_sync(warp_size)

    return warp_idx_sync_kernel


@tilelang.jit(out_idx=[-1])
def _get_warp_idx_kernel(num_threads: int = 128, warp_size: Optional[int] = None):
    @T.prim_func
    def warp_idx_kernel(A: T.Tensor((num_threads,), T.int32)):
        with T.Kernel(1, threads=num_threads) as _:
            tx = T.get_thread_binding()
            A[tx] = T.get_warp_idx(warp_size)

    return warp_idx_kernel


@tilelang.jit(out_idx=[-1])
def _get_warp_group_idx_kernel(
    num_threads: int = 128,
    warp_size: Optional[int] = None,
    warps_per_group: Optional[int] = None,
):
    @T.prim_func
    def warp_group_idx_kernel(A: T.Tensor((num_threads,), T.int32)):
        with T.Kernel(1, threads=num_threads) as _:
            tx = T.get_thread_binding()
            A[tx] = T.get_warp_group_idx(warp_size, warps_per_group)

    return warp_group_idx_kernel


@tilelang.jit(out_idx=[-1])
def _shuffle_elect_kernel(num_threads: int = 128, thread_extent: int = 64):
    @T.prim_func
    def shuffle_elect_kernel(A: T.Tensor((num_threads,), T.int32)):
        with T.Kernel(1, threads=num_threads) as _:
            tx = T.get_thread_binding()
            elected = T.shuffle_elect(thread_extent)
            A[tx] = elected

    return shuffle_elect_kernel


def run_get_lane_id(num_threads: int = 128, warp_size: Optional[int] = None):
    kernel = _get_laneid_kernel(num_threads, warp_size)
    A = kernel()
    print(kernel.get_kernel_source())
    print(A)
    expected_warp_size = _resolve_warp_size(warp_size)
    ref = torch.arange(num_threads, dtype=A.dtype, device=A.device) % expected_warp_size
    torch.testing.assert_close(A.cpu(), ref.cpu())
    return A


def run_get_warp_idx_sync(num_threads: int = 128, warp_size: Optional[int] = None):
    kernel = _get_warp_idx_sync_kernel(num_threads, warp_size)
    A = kernel()
    print(kernel.get_kernel_source())
    print(A)
    expected_warp_size = _resolve_warp_size(warp_size)
    ref = torch.arange(num_threads, dtype=A.dtype, device=A.device) // expected_warp_size
    torch.testing.assert_close(A.cpu(), ref.cpu())
    return A


def run_get_warp_idx(num_threads: int = 128, warp_size: Optional[int] = None):
    kernel = _get_warp_idx_kernel(num_threads, warp_size)
    A = kernel()
    print(kernel.get_kernel_source())
    print(A)
    expected_warp_size = _resolve_warp_size(warp_size)
    ref = torch.arange(num_threads, dtype=A.dtype, device=A.device) // expected_warp_size
    torch.testing.assert_close(A.cpu(), ref.cpu())
    return A


def run_get_warp_group_idx(
    num_threads: int = 128,
    warp_size: Optional[int] = None,
    warps_per_group: Optional[int] = None,
):
    kernel = _get_warp_group_idx_kernel(num_threads, warp_size, warps_per_group)
    A = kernel()
    print(kernel.get_kernel_source())
    print(A)
    expected_warp_size = _resolve_warp_size(warp_size)
    expected_warps_per_group = _resolve_warps_per_group(warps_per_group)
    threads_per_group = expected_warp_size * expected_warps_per_group
    if threads_per_group <= 0:
        raise ValueError("threads_per_group must be positive.")
    ref = torch.arange(num_threads, dtype=A.dtype, device=A.device) // threads_per_group
    torch.testing.assert_close(A.cpu(), ref.cpu())
    return A


def run_shuffle_elect(num_threads: int = 128, thread_extent: int = 64):
    if thread_extent < 0:
        raise ValueError("thread_extent must be non-negative.")
    kernel = _shuffle_elect_kernel(num_threads, thread_extent)
    A = kernel()
    print(kernel.get_kernel_source())
    print(A)
    indices = torch.arange(num_threads, device=A.device, dtype=torch.int64)
    if thread_extent == 0:
        mask = indices == 0
    elif thread_extent > 0:
        mask = (indices % thread_extent) == 0
    else:
        mask = torch.zeros_like(indices, dtype=torch.bool)
    ref = mask.to(dtype=A.dtype, device=A.device)
    torch.testing.assert_close(A.cpu(), ref.cpu())
    return A


@tilelang.testing.requires_cuda
def test_get_lane_idx_default():
    run_get_lane_id()


@tilelang.testing.requires_cuda
def test_get_lane_idx_custom():
    run_get_lane_id(num_threads=256, warp_size=64)


@tilelang.testing.requires_cuda
def test_get_warp_idx_sync_default():
    run_get_warp_idx_sync()


@tilelang.testing.requires_cuda
def test_get_warp_idx_sync_custom():
    run_get_warp_idx_sync(num_threads=256, warp_size=16)


@tilelang.testing.requires_cuda
def test_get_warp_idx_default():
    run_get_warp_idx()


@tilelang.testing.requires_cuda
def test_get_warp_idx_custom():
    run_get_warp_idx(num_threads=320, warp_size=20)


@tilelang.testing.requires_cuda
def test_get_warp_group_idx_default():
    run_get_warp_group_idx()


@tilelang.testing.requires_cuda
def test_get_warp_group_idx_custom():
    run_get_warp_group_idx(num_threads=512, warp_size=32, warps_per_group=5)


@tilelang.testing.requires_cuda
def test_shuffle_elect_default():
    run_shuffle_elect(num_threads=256, thread_extent=64)


@tilelang.testing.requires_cuda
def test_shuffle_elect_block_leader():
    run_shuffle_elect(num_threads=128, thread_extent=0)


if __name__ == "__main__":
    tilelang.testing.main()
