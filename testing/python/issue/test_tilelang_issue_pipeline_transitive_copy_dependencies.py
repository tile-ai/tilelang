import tilelang
import tilelang.language as T
import tilelang.testing
import torch
import pytest


@T.prim_func
def _producer_bind_kernel(
    A: T.Tensor((64,), T.float16),
    B: T.Tensor((64,), T.float16),
):
    with T.Kernel(1, threads=256):
        index_shared = T.alloc_shared((1,), T.int32)
        A_shared = T.alloc_shared((32,), T.float16)
        for k in T.Pipelined(2, num_stages=2):
            index_shared[0] = k * 32
            offset: T.int32 = index_shared[0]
            T.copy(A[offset], A_shared)
            for i in T.Parallel(32):
                B[k * 32 + i] = A_shared[i]


@tilelang.testing.requires_cuda
def test_pipeline_planning_tracks_transitive_copy_index_producer():
    target = tilelang.tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    with target:
        tilelang.lower(_producer_bind_kernel, target=target)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("steps", [2, 5])
def test_pipeline_cross_thread_copy_index_producer_runtime(steps):
    length = steps * 32

    @T.prim_func
    def main(A: T.Tensor((length,), T.float32), B: T.Tensor((length,), T.float32)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            indices = T.alloc_shared((64,), T.int32)
            data = T.alloc_shared((32,), T.float32)
            indices[tx] = 0
            indices[tx + 32] = 0
            for k in T.Pipelined(steps, num_stages=2):
                indices[tx] = k * 32
                indices[tx + 32] = k * 32
                offset: T.int32 = indices[tx + 1]
                data[tx] = A[offset + tx]
                B[k * 32 + tx] = data[tx]

    kernel = tilelang.compile(main, target="cuda")
    a = torch.arange(length, dtype=torch.float32, device="cuda")
    b = torch.full_like(a, -1)
    kernel(a, b)
    torch.testing.assert_close(b, a, atol=0, rtol=0)


@tilelang.testing.requires_cuda
def test_scheduled_bind_with_replayed_input_runtime():
    @T.prim_func
    def main(A: T.Tensor((128,), T.float32), Ids: T.Tensor((4,), T.float32), B: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((32,), T.float32)
            for k in T.serial(
                4,
                annotations={
                    "software_pipeline_stage": [0, 1, 1],
                    "software_pipeline_order": [0, 1, 2],
                    "software_pipeline_replayable_scalar_binds": [1, 0, 0, 0],
                },
            ):
                bias: T.float32 = Ids[k]
                shared[tx] = A[k * 32 + tx]
                value: T.float32 = shared[tx] + bias
                B[k * 32 + tx] = value

    mod = tilelang.transform.InjectSoftwarePipeline()(tilelang.tvm.IRModule.from_expr(main))
    kernel = tilelang.compile(mod["main"], target="cuda")
    a = torch.arange(128, dtype=torch.float32, device="cuda")
    ids = torch.arange(4, dtype=torch.float32, device="cuda") * 100
    b = torch.full_like(a, -1)
    kernel(a, ids, b)
    torch.testing.assert_close(b.view(4, 32), a.view(4, 32) + ids[:, None], atol=0, rtol=0)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9)
@pytest.mark.parametrize("num_stages", [0, 2])
def test_transitive_tma_copy_falls_back_to_sequential(num_stages):
    @T.prim_func
    def main(Data: T.Tensor((1, 8, 8, 32), T.float16), A: T.Tensor((32,), T.float16), B: T.Tensor((64,), T.float16)):
        with T.Kernel(1, threads=128):
            patch = T.alloc_shared((16, 32), T.float16)
            shared = T.alloc_shared((32,), T.float16)
            for k in T.Pipelined(2, num_stages=num_stages):
                T.im2col(Data, patch, 0, k, 3, 1, 1, 1)
                offset: T.int32 = T.cast(patch[0, 0], T.int32)
                T.copy(A[offset], shared)
                for i in T.Parallel(32):
                    B[k * 32 + i] = shared[i]

    kernel = tilelang.compile(main, target="cuda", pass_configs={"tl.disable_warp_specialized": True})
    data = torch.zeros((1, 8, 8, 32), dtype=torch.float16, device="cuda")
    a = torch.arange(32, dtype=torch.float16, device="cuda")
    b = torch.full((64,), -1, dtype=torch.float16, device="cuda")
    kernel(data, a, b)
    torch.testing.assert_close(b, a.repeat(2), atol=0, rtol=0)


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("steps", [2, 5])
def test_dependent_copy_waits_for_async_index_runtime(steps):
    length = steps * 32

    @T.prim_func
    def main(Ids: T.Tensor((steps,), T.int32), A: T.Tensor((length,), T.float32), B: T.Tensor((length,), T.float32)):
        with T.Kernel(1, threads=32):
            index = T.alloc_shared((1,), T.int32)
            shared = T.alloc_shared((32,), T.float32)
            for k in T.Pipelined(steps, num_stages=2):
                T.copy(Ids[k], index)
                T.copy(A[index[0]], shared)
                for i in T.Parallel(32):
                    B[k * 32 + i] = shared[i]

    kernel = tilelang.compile(main, target="cuda", pass_configs={"tl.disable_warp_specialized": True})
    ids = torch.arange(steps - 1, -1, -1, dtype=torch.int32, device="cuda") * 32
    a = torch.arange(length, dtype=torch.float32, device="cuda")
    b = torch.full_like(a, -1)
    kernel(ids, a, b)
    torch.testing.assert_close(b.view(steps, 32), a.view(steps, 32).flip(0), atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
