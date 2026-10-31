"""Regression tests for liveness regions in MergeSharedMemoryAllocations.

The liveness analysis represents buffer lifetimes as linear intervals over
the flattened statement sequence. A loop body appears only once in that
sequence, so "linearly disjoint" does not imply "dynamically disjoint" once
back edges are involved. The kill-reorder rule must therefore reason about
loop regions, not scope levels or linear positions:

1. ``test_no_cross_iteration_overwrite_across_while_back_edge``: a buffer
   loaded before a While loop and read in every iteration is live across
   the back edge. Its kill must be extended to the loop's dynamic exit so
   the arena packer refuses to overlap it with a buffer first written
   inside the same loop. Reusing there overwrites the loop-invariant
   buffer across iterations and silently corrupts the output.

2. ``test_sequential_loop_regions_still_reuse``: two sequential loops are
   distinct regions. A buffer live only inside the second loop must still
   reuse the region of a buffer whose lifetime ends with the first loop;
   extending the first buffer's kill past the loop pair would fabricate an
   overlap and refuse a dynamically legal reuse.
"""

import re

import tilelang
import tilelang.language as T
import tilelang.testing


def _merge_shared_pass_configs():
    return {
        # Merge the static ("shared"-scope) allocations into one arena.
        "tirx.merge_static_smem": True,
        # Attribute every touch to its deepest scope so in-loop events are
        # visible to the liveness analysis.
        "tl.enable_aggressive_shared_memory_merge": True,
    }


def _shared_arena_offsets(src):
    """Map each merged shared buffer to its byte offset in the arena.

    The CUDA backend materializes the merged allocations as dynamic shared
    memory: ``void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));``.
    """
    return {name: int(off) for name, off in re.findall(r"void\* (\w+) = \(\(void\*\)\(\(char\*\)buf_dyn_shmem \+ (\d+)\)\)", src)}


@tilelang.testing.requires_cuda
def test_no_cross_iteration_overwrite_across_while_back_edge():
    N = 64
    NITER = 4

    @T.prim_func
    def kernel(A: T.Tensor((N,), T.float16), C: T.Tensor((N,), T.float16)):
        with T.Kernel(1, threads=128):
            # Q geometry: loaded once before the loop, read every iteration.
            A_shared = T.alloc_shared((N,), T.float16)
            # acc geometry: first written inside the loop.
            acc_shared = T.alloc_shared((N,), T.float16)
            x = T.alloc_fragment((N,), T.float16)
            cnt = T.alloc_fragment((1,), T.int32)

            for i in T.serial(N):
                A_shared[i] = A[i]
            cnt[0] = 0
            while cnt[0] < NITER:
                for i in T.serial(N):
                    x[i] = A_shared[i]
                for i in T.serial(N):
                    acc_shared[i] = x[i] * 2.0
                cnt[0] = cnt[0] + 1
            for i in T.serial(N):
                C[i] = acc_shared[i]

    func = tilelang.compile(kernel, out_idx=[1], pass_configs=_merge_shared_pass_configs())

    # Structural check: the packer must judge A_shared (live across the
    # back edge) as overlapping acc_shared (first written inside the loop),
    # so the two 128-byte buffers get disjoint regions in the arena.
    offsets = _shared_arena_offsets(func.get_kernel_source())
    assert {"A_shared", "acc_shared"} <= set(offsets), (
        f"merged dynamic shared arena (buf_dyn_shmem) not found; buffers seen: {sorted(offsets)}"
    )
    assert sorted(offsets.values()) == [0, 128], f"expected two disjoint 128-byte regions (no reuse across the back edge); got {offsets}"

    # Numerical check: if the packer wrongly treated A_shared as dead
    # before acc_shared's first write, acc_shared would reuse A_shared's
    # region; the next iteration would then read the overwritten values
    # and the result would double every round (2A -> 4A -> 8A -> 16A).
    import torch

    A = (torch.arange(N, dtype=torch.float32) + 1).half().cuda()
    C = func(A)
    if isinstance(C, (tuple, list)):
        C = C[0]
    torch.testing.assert_close(C, A * 2)


@tilelang.testing.requires_cuda
def test_sequential_loop_regions_still_reuse():
    N = 64
    NITER = 4

    @T.prim_func
    def kernel(A: T.Tensor((N,), T.float16), C: T.Tensor((N,), T.float16)):
        with T.Kernel(1, threads=128):
            # Read every iteration of the first loop, dead at its exit.
            S = T.alloc_shared((N,), T.float16)
            # Live only inside the second (sequential) loop.
            Tmp = T.alloc_shared((N,), T.float16)
            x = T.alloc_fragment((N,), T.float16)
            w = T.alloc_fragment((N,), T.float32)

            for i in T.serial(N):
                S[i] = A[i]
            T.clear(w)
            for _t in T.serial(NITER):
                for i in T.serial(N):
                    x[i] = S[i]
                for i in T.serial(N):
                    w[i] = w[i] + 1.0
            for _t in T.serial(NITER):
                for i in T.serial(N):
                    Tmp[i] = x[i] * 2.0
                for i in T.serial(N):
                    C[i] = Tmp[i] + w[i]

    func = tilelang.compile(kernel, out_idx=[1], pass_configs=_merge_shared_pass_configs())

    # Structural check: the two loops are sequential regions with no back
    # edge between them, so Tmp must reuse S's region instead of being
    # packed after it.
    offsets = _shared_arena_offsets(func.get_kernel_source())
    assert {"S", "Tmp"} <= set(offsets), f"merged dynamic shared arena (buf_dyn_shmem) not found; buffers seen: {sorted(offsets)}"
    assert offsets["S"] == offsets["Tmp"] == 0, f"expected Tmp to reuse S's region at offset 0; got {offsets}"

    # Numerical check: C = 2 * A + NITER.
    import torch

    A = (torch.arange(N, dtype=torch.float32) + 1).half().cuda()
    C = func(A)
    if isinstance(C, (tuple, list)):
        C = C[0]
    torch.testing.assert_close(C, A * 2 + NITER)


if __name__ == "__main__":
    tilelang.testing.main()
