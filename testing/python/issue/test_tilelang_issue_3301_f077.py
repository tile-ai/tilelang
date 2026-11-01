"""Regression test for GitHub issue #3301 (F077).

#3301 F077 - T.Pipelined gathers a stale index for an indexed T.copy when the
index is produced across the loop back-edge.
"""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def _kernel(loop, block=64, threads=256, fill=False):
    @T.prim_func
    def main(A: T.Tensor((block * 4,), T.float16), B: T.Tensor((block * 4,), T.float16)):
        with T.Kernel(1, threads=threads):
            index_shared = T.alloc_shared((1,), T.int32)
            a_shared = T.alloc_shared((block,), T.float16)
            index_shared[0] = 0
            for k in loop(4):
                offset = index_shared[0]
                T.copy(A[offset : offset + block], a_shared)
                for i in T.Parallel(block):
                    B[k * block + i] = a_shared[i]
                if fill:
                    T.fill(index_shared, (k + 1) * block)
                else:
                    index_shared[0] = (k + 1) * block

    return main


def _run(kernel, size):
    fn = tilelang.compile(kernel, out_idx=[1])
    a = torch.arange(size, dtype=torch.float16, device="cuda")
    out = fn(a)
    torch.cuda.synchronize()
    torch.testing.assert_close(out, a, rtol=0, atol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
@pytest.mark.parametrize("num_stages", [1, 2])
def test_pipelined_copy_index_from_previous_iteration(num_stages):
    _run(_kernel(lambda n: T.Pipelined(n, num_stages=num_stages)), 256)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9, 0)
def test_pipelined_copy_index_with_tileop_write():
    _run(_kernel(lambda n: T.Pipelined(n, num_stages=1), fill=True), 256)


@tilelang.testing.requires_cuda
def test_pipelined_copy_index_without_tma():
    # A 4-element copy is too small for TMA and takes the software pipeline.
    _run(_kernel(lambda n: T.Pipelined(n, num_stages=1), block=4, threads=128), 16)


@tilelang.testing.requires_cuda
def test_explicit_schedule_with_loop_carried_copy():
    def explicit(stage):
        return _kernel(lambda n: T.Pipelined(n, order=[0, 1, 2, 3], stage=stage), block=4, threads=128)

    # The store runs a stage after the copy that reads it.
    with pytest.raises(Exception, match="loop-carried"):
        tilelang.compile(explicit([0, 0, 0, 1]), out_idx=[1])
    # The store stays in the copy's stage, after it.
    _run(explicit([0, 0, 1, 0]), 16)


if __name__ == "__main__":
    tilelang.testing.main()
