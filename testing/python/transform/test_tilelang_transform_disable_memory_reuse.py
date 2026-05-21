"""Tests for TL_DISABLE_SHARED_MEMORY_REUSE pass config.

When `TL_DISABLE_SHARED_MEMORY_REUSE` is True, shared memory allocations are still
merged into a single buffer, but each buffer gets its own dedicated region without
lifetime-based reuse (i.e., no two buffers share the same offset even if their
lifetimes don't overlap).
"""

import re

import torch
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import PassConfigKey

N = 1024


def _make_data_integrity_kernel():

    @tilelang.jit(
        pass_configs={
            PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: True,
        },
    )
    def kernel(A, B, C, D):
        A: T.Tensor[[N], T.float16]
        B: T.Tensor[[N], T.float16]
        C: T.Tensor[[N], T.float16]
        D: T.Tensor[[N], T.float16]

        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared([N], T.float16)
            b_shared = T.alloc_shared([N], T.float16)
            c_shared = T.alloc_shared([N], T.float16)
            d_frag = T.alloc_fragment([N], T.float16)

            T.copy(A, a_shared)
            T.copy(B, b_shared)
            T.copy(C, c_shared)

            for i in T.Parallel(N):
                d_frag[i] = a_shared[i] + b_shared[i] + c_shared[i]

            T.copy(d_frag, D)

    return kernel


def _make_no_overlap_kernel(disable_reuse: bool):

    @tilelang.jit(
        pass_configs={
            PassConfigKey.TL_DISABLE_SHARED_MEMORY_REUSE: disable_reuse,
        },
    )
    def kernel(A, B, A_out, B_out):
        A: T.Tensor[[N], T.float16]
        B: T.Tensor[[N], T.float16]
        A_out: T.Tensor[[N], T.float16]
        B_out: T.Tensor[[N], T.float16]

        with T.Kernel(1, threads=128):
            a_shared = T.alloc_shared([N], T.float16)
            T.copy(A, a_shared)
            T.copy(a_shared, A_out)

            b_shared = T.alloc_shared([N], T.float16)
            T.copy(B, b_shared)
            T.copy(b_shared, B_out)

    return kernel


@tilelang.testing.requires_cuda
def test_disable_reuse_data_integrity():
    """Allocate multiple shared buffers, copy data in, compute sum, copy out.

    Verifies data integrity when shared memory reuse is disabled.
    """
    kernel = _make_data_integrity_kernel()

    a = torch.randn(N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)
    c = torch.randn(N, device="cuda", dtype=torch.float16)
    d = torch.empty(N, device="cuda", dtype=torch.float16)

    kernel(a, b, c, d)
    ref = a + b + c
    torch.testing.assert_close(d, ref, rtol=1e-3, atol=1e-3)


@tilelang.testing.requires_cuda
def test_disable_reuse_no_overlap():
    """Two sequential buffers must NOT share the same offset when reuse is disabled.

    a_shared is used (copy in, copy out), then b_shared is used (copy in, copy out).
    Their lifetimes don't overlap, so with reuse enabled they share offset 0.
    With reuse disabled, b_shared must get a different (non-zero) offset.
    """
    kernel_no_reuse = _make_no_overlap_kernel(disable_reuse=True)
    kernel_reuse = _make_no_overlap_kernel(disable_reuse=False)

    a = torch.randn(N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)
    a_out = torch.empty(N, device="cuda", dtype=torch.float16)
    b_out = torch.empty(N, device="cuda", dtype=torch.float16)

    # Correctness: both should produce correct results
    kernel_no_reuse(a, b, a_out, b_out)
    torch.testing.assert_close(a_out, a, rtol=0, atol=0)
    torch.testing.assert_close(b_out, b, rtol=0, atol=0)

    a_out2 = torch.empty(N, device="cuda", dtype=torch.float16)
    b_out2 = torch.empty(N, device="cuda", dtype=torch.float16)
    kernel_reuse(a, b, a_out2, b_out2)
    torch.testing.assert_close(a_out2, a, rtol=0, atol=0)
    torch.testing.assert_close(b_out2, b, rtol=0, atol=0)

    # Verify shared memory layout difference:
    # With reuse: both buffers use offset 0 in buf_dyn_shmem.
    # Without reuse: second buffer gets a separate region (non-zero offset).
    src_no_reuse = kernel_no_reuse.get_kernel_source()
    src_reuse = kernel_reuse.get_kernel_source()

    def extract_smem_offsets(src: str) -> list[int]:
        """Extract merged shared-memory offsets from generated code."""
        # Alias-preserving lowering emits:
        #   void* a_shared = ((void*)((char*)buf_dyn_shmem + 0));
        alias_pattern = r"void\*\s+\w+\s*=\s*\(\(void\*\)\(\(char\*\)buf_dyn_shmem\s*\+\s*(\d+)\)\);"
        # Direct merged-buffer lowering emits access patterns such as:
        #   buf_dyn_shmem)[1024])
        direct_pattern = r"buf_dyn_shmem\)\[(\d+)\]"
        offsets = {int(m) for m in re.findall(alias_pattern, src)}
        offsets.update(int(m) for m in re.findall(direct_pattern, src))
        return sorted(offsets)

    offsets_no_reuse = extract_smem_offsets(src_no_reuse)
    offsets_reuse = extract_smem_offsets(src_reuse)

    # With reuse disabled: must have at least 2 distinct offsets (buffers not merged)
    assert len(offsets_no_reuse) >= 2, f"Expected >=2 distinct smem offsets with reuse disabled, got {offsets_no_reuse}"

    # With reuse enabled: should have only 1 offset (buffers share the same region)
    assert len(offsets_reuse) == 1, f"Expected 1 smem offset with reuse enabled, got {offsets_reuse}"


if __name__ == "__main__":
    test_disable_reuse_data_integrity()
    test_disable_reuse_no_overlap()
    print("All tests passed!")
