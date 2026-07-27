"""Regression test for GitHub issue #2188.

#2188 - Feature: native T.named_barrier_arrive (bar.arrive) API.
"""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.testing.requires_cuda
def test_named_barrier_arrive_api():
    """T.named_barrier_arrive emits the CUDA named-barrier arrive helper."""

    @tilelang.jit(out_idx=[1])
    def kernel_fn():
        @T.prim_func
        def kernel(inp: T.Tensor((1,), T.int32), out: T.Tensor((1,), T.int32)):
            with T.Kernel(1, threads=64) as _:
                tx = T.get_thread_binding()
                slot = T.alloc_shared((1,), T.int32)

                ready_barrier = 10
                empty_barrier = 11
                participant_threads = 64

                if tx < 32:
                    if tx == 0:
                        slot[0] = inp[0] + 1
                    T.named_barrier_arrive(ready_barrier, participant_threads)
                    T.sync_threads(empty_barrier, participant_threads)
                else:
                    T.sync_threads(ready_barrier, participant_threads)
                    if tx == 32:
                        out[0] = slot[0]
                    T.named_barrier_arrive(empty_barrier, participant_threads)

        return kernel

    jit_kernel = kernel_fn()
    source = jit_kernel.get_kernel_source()

    assert "tl::__named_barrier_arrive(10, 64);" in source
    assert "tl::__named_barrier_arrive(11, 64);" in source

    inp = torch.tensor([41], device="cuda", dtype=torch.int32)
    out = jit_kernel(inp)
    torch.testing.assert_close(out.cpu(), torch.tensor([42], dtype=torch.int32))


if __name__ == "__main__":
    test_named_barrier_arrive_api()
