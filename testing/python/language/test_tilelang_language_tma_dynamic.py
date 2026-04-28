import tilelang
import tilelang.testing
from tilelang import language as T
import pytest
import torch


@tilelang.jit
def make_dynamic_copy_kernel():
    @T.prim_func
    def kernel(
        A: T.Tensor([128, 128], T.float16),
        B: T.Tensor([128, 128], T.float16),
        actual_rows: T.int32,
    ):
        with T.Kernel(1, threads=128) as bx:
            shared = T.alloc_shared([128, 128], T.float16)
            # Dynamic M-dim extent: should trigger tensormap.replace
            T.copy(A[:actual_rows, :], shared[:actual_rows, :])
            T.copy(shared[:actual_rows, :], B[:actual_rows, :])

    return kernel


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_dynamic_tma_copy_emits_tensormap_replace():
    """T.copy with dynamic extents emits tensormap.replace + fence sequence."""
    kernel = make_dynamic_copy_kernel()

    A = torch.randn((128, 128), dtype=torch.float16, device='cuda')
    B = torch.zeros_like(A)
    kernel(A, B, 127)
    print(kernel.get_kernel_source())
    assert torch.allclose(B[:127, :], A[:127, :])
    assert torch.allclose(B[127:, :], torch.zeros_like(A[127:, :]))

    src = kernel.get_kernel_source()
    assert "tensormap_copy_to_smem" in src, (
        "Expected tensormap_copy_to_smem for dynamic extents:\n" + src
    )
    assert "tensormap_replace_box_dim" in src, (
        "Expected tensormap_replace_box_dim for dynamic extents:\n" + src
    )
    assert "tensormap_cp_fence_release" in src, (
        "Expected tensormap_cp_fence_release for dynamic extents:\n" + src
    )
    assert "tensormap_fence_acquire" in src, (
        "Expected tensormap_fence_acquire for dynamic extents:\n" + src
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_dynamic_copy_with_load_requires_barrier():
    """T.tma_copy load with dynamic extent requires barrier (unchanged)."""
    @T.prim_func
    def kernel(
        A: T.Tensor([4096, block_K], T.float16),
        actual_rows: T.int32,
    ):
        with T.Kernel(1, threads=128) as bx:
            A_shared = T.alloc_shared([block_M, block_K], T.float16)
            mbar = T.alloc_barrier(1)
            T.tma_copy(A[:actual_rows, :], A_shared[:actual_rows, :],
                       barrier=mbar[0])
    src = tilelang.compile(kernel).get_kernel_source()
    assert "tensormap_replace_box_dim" in src, (
        "TMA load with dynamic extent should also emit tensormap.replace:\n" + src
    )


if __name__ == "__main__":
    test_dynamic_tma_copy_emits_tensormap_replace()
    # test_static_tma_copy_no_tensormap_replace()
    # test_dynamic_copy_with_load_requires_barrier()
    # print("All tests passed!")
