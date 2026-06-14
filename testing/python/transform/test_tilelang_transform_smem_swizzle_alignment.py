"""Regression tests for TMA copies into swizzled shared memory.

TMA applies the descriptor swizzle relative to the destination shared-memory
base. A swizzled destination therefore needs to start on the swizzle period
boundary, not just on the generic 128B TMA base alignment.
"""

import re

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_swizzled_layout

TILE_M = 128
THREADS = 256
INT32_BYTES = 4
MISALIGNING_PAD_ELEMS = 32


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True})
def _make_swizzled_tma_copy_kernel(pad_elems: int, dtype: str, columns: int):
    elems_per_thread = TILE_M * columns // THREADS

    @T.prim_func
    def kernel(
        A: T.Tensor[(TILE_M, columns), dtype],
        out: T.Tensor[(TILE_M, columns), dtype],
        dummy: T.Tensor[(1,), T.int32],
    ):
        with T.Kernel(1, threads=THREADS) as _:
            pad = T.alloc_shared((pad_elems,), T.int32)
            smem = T.alloc_shared((TILE_M, columns), dtype)
            T.annotate_layout({smem: make_swizzled_layout(smem)})

            tid = T.get_thread_binding(0)
            if tid < pad_elems:
                pad[tid] = tid
            T.sync_threads()

            bar = T.alloc_barrier(THREADS)
            T.tma_copy(A[0, 0], smem, barrier=bar)
            T.barrier_arrive(bar)
            T.mbarrier_wait_parity(bar, 0)
            for i in T.serial(elems_per_thread):
                elem = tid * elems_per_thread + i
                out[elem // columns, elem % columns] = smem[elem // columns, elem % columns]

            if tid == 0:
                dummy[0] = pad[pad_elems - 1]

    return kernel


def _compile_kernel(pad_elems: int, dtype: str, columns: int):
    tilelang.disable_cache()
    try:
        return _make_swizzled_tma_copy_kernel(pad_elems, dtype, columns)
    finally:
        tilelang.enable_cache()


def _dynamic_smem_offset(source: str, buffer_name: str) -> int:
    """Extract the byte offset of a dynamic shared-memory alias."""
    pattern = rf"void\*\s+{re.escape(buffer_name)}\s*=\s*\(\(void\*\)\(\(char\*\)buf_dyn_shmem\s*\+\s*(\d+)\)\)"
    match = re.search(pattern, source)
    assert match is not None, f"dynamic shared buffer {buffer_name!r} not found in source:\n{source}"
    return int(match.group(1))


SWIZZLE_ALIGNMENT_CASES = [
    pytest.param("float16", 16, 256, id="32B-swizzle"),
    pytest.param("float16", 32, 512, id="64B-swizzle"),
    pytest.param("float16", 64, 1024, id="128B-swizzle"),
]

PAD_LIVENESS_CASES = [
    pytest.param(32, id="pad-128B"),
    pytest.param(96, id="pad-384B"),
    pytest.param(256, id="pad-1024B"),
]


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize(("dtype", "columns", "required_alignment"), SWIZZLE_ALIGNMENT_CASES)
def test_tma_swizzled_smem_uses_swizzle_period_alignment(dtype, columns, required_alignment):
    # 32 int32 values create a 128B live allocation before smem. That is enough
    # for the generic TMA requirement, but not enough for any swizzle period.
    kernel = _compile_kernel(MISALIGNING_PAD_ELEMS, dtype, columns)
    source = kernel.get_kernel_source()
    pad_offset = _dynamic_smem_offset(source, "pad")
    smem_offset = _dynamic_smem_offset(source, "smem")

    assert pad_offset == 0
    assert MISALIGNING_PAD_ELEMS * INT32_BYTES == 128
    assert smem_offset % required_alignment == 0, (
        f"swizzled smem tile at +{smem_offset}B violates required {required_alignment}B alignment\n{source}"
    )


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
@pytest.mark.parametrize("pad_elems", PAD_LIVENESS_CASES)
def test_tma_swizzled_copy_correctness(pad_elems):
    import torch

    columns = 64
    kernel = _compile_kernel(pad_elems, "bfloat16", columns)
    A = torch.randn((TILE_M, columns), dtype=torch.bfloat16, device="cuda")
    out = torch.zeros_like(A)
    dummy = torch.zeros((1,), dtype=torch.int32, device="cuda")

    kernel(A, out, dummy)
    torch.cuda.synchronize()

    torch.testing.assert_close(out, A, rtol=0, atol=0)
    # Keep the pad live so the allocator has to place smem after it.
    assert int(dummy.item()) == pad_elems - 1


if __name__ == "__main__":
    tilelang.testing.main()
