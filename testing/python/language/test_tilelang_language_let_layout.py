"""
Test layout inference for LetStmt expressions.

This test validates that TileLang correctly handles layout inference when
fragment buffer accesses occur through let bindings. For example:

    block_mask_f = T.alloc_fragment((N_S,), T.int32)
    T.copy(BlockMask[by, :], block_mask_f)
    for i in T.Pipelined(N_S):
        a = block_mask_f[i]  # LetStmt: a is bound to fragment buffer load
        T.copy(A[a, 0], A_shared)  # a is used as index in TMA copy

Key scenarios tested:
1. Fragment buffer layout inference through let bindings
2. TMA (Tensor Memory Accelerator) copy with let-bound indices
3. CP.ASYNC copy with let-bound indices
4. Warp specialization with let-bound fragment accesses
"""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch


def blocksparse_copy_kernel(M, N, N_S, block_M, block_N, dtype=T.float16):
    """BlockSparse copy kernel using fragment for block mask indices."""
    block_mask_shape = (M // block_M, N_S)

    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        B: T.Tensor((M, N), dtype),
        BlockMask: T.Tensor(block_mask_shape, T.int32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)
            block_mask_f = T.alloc_fragment((N_S,), T.int32)

            T.clear(B_shared)
            T.copy(BlockMask[by, :], block_mask_f)
            for i in T.Pipelined(N_S):
                a = block_mask_f[i]  # LetStmt: fragment buffer access
                if a >= 0:
                    T.copy(A[a, 0], A_shared)
                    T.copy(A_shared, B[by * block_M : (by + 1) * block_M, i * block_N : (i + 1) * block_N])

    return main


def ref_blocksparse_copy(A, B, BlockMask, M, N, N_S, block_M, block_N):
    """Reference implementation for blocksparse copy."""
    ref_B = B.clone()
    num_row_blocks = M // block_M

    for by in range(num_row_blocks):
        for i in range(N_S):
            src_row_start = BlockMask[by, i].item()
            ref_B[by * block_M : (by + 1) * block_M, i * block_N : (i + 1) * block_N] = A[
                src_row_start : src_row_start + block_M, 0:block_N
            ]

    return ref_B


def run_blocksparse_copy(M, N, block_M, block_N, pass_configs=None):
    """Run blocksparse copy test with given parameters."""
    N_S = N // block_N

    program = blocksparse_copy_kernel(M, N, N_S, block_M, block_N)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        target="cuda",
        pass_configs=pass_configs or {},
    )

    # Initialize tensors
    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    # Create BlockMask with valid row indices
    num_row_blocks = M // block_M
    block_mask = torch.zeros((num_row_blocks, N_S), dtype=torch.int32, device="cuda")
    for by in range(num_row_blocks):
        for i in range(N_S):
            max_row_block = (M - block_M) // block_M
            block_mask[by, i] = torch.randint(0, max_row_block + 1, (1,)).item() * block_M

    # Run kernel
    c = kernel(a, block_mask)

    # Compute reference
    ref_c = ref_blocksparse_copy(a, b, block_mask, M, N, N_S, block_M, block_N)

    # Verify
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_cuda
def test_blocksparse_copy_tma():
    """Test blocksparse copy with TMA (Tensor Memory Accelerator)."""
    run_blocksparse_copy(M=1024, N=1024, block_M=128, block_N=128, pass_configs={})


@tilelang.testing.requires_cuda
def test_blocksparse_copy_cp_async():
    """Test blocksparse copy with CP.ASYNC (without TMA)."""
    run_blocksparse_copy(
        M=1024,
        N=1024,
        block_M=128,
        block_N=128,
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )


if __name__ == "__main__":
    tilelang.testing.main()
