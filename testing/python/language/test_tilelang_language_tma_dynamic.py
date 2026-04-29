import tilelang
from tilelang import language as T
import torch


def make_dynamic_copy_kernel():
    @T.prim_func
    def kernel(
        A: T.Tensor([128, 128], T.float16),
        B: T.Tensor([128, 128], T.float16),
        actual_rows: T.int32,
    ):
        with T.Kernel(1, threads=128) as bx:
            shared = T.alloc_shared([128, 128], T.float16)
            T.copy(A[:actual_rows, :], shared[:actual_rows, :])
            T.copy(shared[:actual_rows, :], B[:actual_rows, :])

    return kernel


def test_dynamic_tma_copy_emits_tensormap_replace():
    """T.copy with dynamic extents emits tensormap.replace + fence sequence."""
    kernel = tilelang.compile(make_dynamic_copy_kernel())
    src = kernel.get_kernel_source()
    print(src)

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
    assert "prefetch_tma_descriptor" in src
    assert "CUtensorMap *tma_workspace" in src

    A = torch.randn((128, 128), dtype=torch.float16, device='cuda')
    B = torch.zeros_like(A)
    kernel(A, B, 127)
    torch.cuda.synchronize()
   
    assert torch.allclose(B[:127, :], A[:127, :])
    assert torch.allclose(B[127:, :], torch.zeros_like(A[127:, :]))


if __name__ == "__main__":
    test_dynamic_tma_copy_emits_tensormap_replace()
