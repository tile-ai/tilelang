"""Regression for #3301 (F077): a pipelined copy reads a loop-carried index."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _kernel(loop, block=64, threads=256, fill=False, via_scratch=False):
    @T.prim_func
    def main(A: T.Tensor((block * 4,), T.float16), B: T.Tensor((block * 4,), T.float16)):
        with T.Kernel(1, threads=threads):
            index_shared = T.alloc_shared((1,), T.int32)
            a_shared = T.alloc_shared((block,), T.float16)
            scratch = T.alloc_shared((1,), T.int32) if via_scratch else None
            index_shared[0] = 0
            for k in loop(4):
                if via_scratch:
                    scratch[0] = index_shared[0]
                offset = scratch[0] if via_scratch else index_shared[0]
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


def _plan(kernel):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    mod = tvm.IRModule.from_expr(kernel.with_attr("target", target))
    return tilelang.transform.PipelinePlanning()(mod)


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
def test_explicit_schedule_with_versioned_copy_buffer():
    @T.prim_func
    def main(A: T.Tensor((16,), T.float16), B: T.Tensor((16,), T.float16)):
        with T.Kernel(1, threads=128):
            scratch = T.alloc_shared((4,), T.float16)
            for k in T.Pipelined(4, order=[0, 1, 2], stage=[0, 1, 1]):
                T.copy(A[k * 4 : k * 4 + 4], scratch)
                offset: T.int32 = T.cast(scratch[0], T.int32)
                T.copy(scratch, B[offset : offset + 4])

    _run(main, 16)


def test_pipeline_planning_tracks_copy_index_through_scratch():
    pipelined = _kernel(lambda n: T.Pipelined(n, num_stages=2), block=4, threads=128, via_scratch=True)
    sequential = _kernel(T.serial, block=4, threads=128, via_scratch=True)
    tvm.ir.assert_structural_equal(_plan(pipelined), _plan(sequential), map_free_vars=True)


def _run_ws_pass(func):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90"})
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    return tilelang.cuda.transform.ProducerConsumerWarpSpecialized()(mod)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("update", [False, True])
def test_ws_tracks_fill_value_in_producer_prefix(update):
    @T.prim_func
    def main(
        A: T.Tensor((256, 64), T.float16),
        Flags: T.Tensor((1,), T.float16),
        Filled: T.Tensor((4, 64), T.float16),
        Out: T.Tensor((256, 64), T.float16),
    ):
        with T.Kernel(1, threads=256):
            S_shared = T.alloc_shared((64, 64), T.float16)
            for ko in T.Pipelined(4, num_stages=2):
                T.fill(Filled[ko, :], Flags[0])
                T.copy(A[ko * 64, 0], S_shared)
                for i, j in T.Parallel(64, 64):
                    Out[ko * 64 + i, j] = S_shared[i, j]
                if update:
                    Flags[0] = T.cast(ko + 1, T.float16)

    script = _run_ws_pass(main)["main"].script()
    assert ("tl_tiled_ws_applied" in script) == (not update)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("update", [False, True])
def test_ws_tracks_tvm_access_ptr_write_to_parameter_buffer(update):
    @T.prim_func
    def main(
        A: T.Tensor((256, 64), T.float16),
        idx: T.Tensor((1,), T.int32),
        Out: T.Tensor((256, 64), T.float16),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((64, 64), T.float16)
            for ko in T.Pipelined(4, num_stages=2):
                T.copy(A[idx[0], 0], A_shared)
                for i, j in T.Parallel(64, 64):
                    Out[ko * 64 + i, j] = A_shared[i, j]
                if update:
                    T.call_extern("int32", "write_index", idx.access_ptr("w"))

    script = _run_ws_pass(main)["main"].script()
    assert ("tl_tiled_ws_applied" in script) == (not update)


if __name__ == "__main__":
    tilelang.testing.main()
