"""Real CUDA/CuTe cluster launches retain per-kernel geometry and host ABI."""

import pytest
import torch
import tilelang
import tilelang.language as T


def _compile(function, backend, monkeypatch, outputs=(1,)):
    cute = backend.startswith("cutedsl_")
    if cute:
        monkeypatch.setenv("TILELANG_CUTEDSL_HOST_LAUNCHER", backend.removeprefix("cutedsl_"))
        monkeypatch.setenv("TILELANG_CUTEDSL_REQUIRE_CUTLASS_HOST", "1")
    return tilelang.compile(
        function,
        out_idx=list(outputs),
        target={"kind": "cutedsl" if cute else "cuda", "arch": "sm_100a"},
        target_host={"kind": "c"},
        execution_backend="cutedsl" if cute else backend,
        pass_configs={"tl.disable_warp_specialized": True},
    )


@pytest.mark.parametrize("backend", ["tvm_ffi", "cython"])
def test_cuda_cluster_launch_matches_hardware_rank(backend, monkeypatch, tmp_path):
    @T.prim_func
    def main(A: T.Tensor((8,), "int32"), B: T.Tensor((8,), "int32")):
        with T.ClusterKernel(8, cluster_dims=2, threads=32) as block:
            T.cluster_arrive()
            T.cluster_wait()
            if T.get_thread_binding() == 0:
                B[block] = A[block] + T.block_rank_in_cluster()
            T.cluster_sync()

    compiled = _compile(main, backend, monkeypatch)
    data = torch.arange(8, dtype=torch.int32, device="cuda")
    torch.testing.assert_close(compiled(data), data + data % 2, rtol=0, atol=0)
    (tmp_path / "device.cu").write_text(compiled.get_kernel_source())


@pytest.mark.parametrize("backend", ["tvm_ffi", "cython", "cutedsl_cpp", "cutedsl_cutlass"])
def test_per_kernel_clusters_dynamic_grids_outputs_and_stream(backend, monkeypatch, tmp_path):
    n = T.dynamic("n", "int64")

    @T.prim_func
    def main(A: T.Tensor((n, 2), "int32"), Intermediate: T.Tensor((n, 2), "int32"), B: T.Tensor((n, 2), "int32")):
        if n > 0:
            with T.ClusterKernel(n, 2, cluster_dims=(2, 1, 1), threads=32) as (bx, by):
                T.cluster_sync()
                if T.get_thread_binding() == 0:
                    Intermediate[bx, by] = A[bx, by] + T.block_rank_in_cluster()
            with T.ClusterKernel(n, 2, cluster_dims=(1, 2, 1), threads=64) as (bx, by):
                T.cluster_sync()
                if T.get_thread_binding() == 0:
                    B[bx, by] = Intermediate[bx, by] + 10 * T.block_rank_in_cluster()
            with T.Kernel(n, 2, threads=32) as (bx, by):
                if T.get_thread_binding() == 0:
                    B[bx, by] += 1

    compiled = _compile(main, backend, monkeypatch, outputs=(2, 1))
    stream = torch.cuda.Stream()
    for size in (0, 2, 8, 18):
        with torch.cuda.stream(stream):
            data = torch.arange(size * 2, dtype=torch.int32, device="cuda").reshape(size, 2)
            final, intermediate = compiled(data)
            first_rank = torch.arange(size, dtype=torch.int32, device="cuda")[:, None] % 2
            second_rank = torch.arange(2, dtype=torch.int32, device="cuda")[None, :] % 2
            expected_intermediate = data + first_rank
            expected = expected_intermediate + 10 * second_rank + 1
        stream.synchronize()
        torch.testing.assert_close(intermediate, expected_intermediate, rtol=0, atol=0)
        torch.testing.assert_close(final, expected, rtol=0, atol=0)
    with pytest.raises((ValueError, RuntimeError), match="divisible"):
        compiled(torch.zeros((3, 2), dtype=torch.int32, device="cuda"))
    (tmp_path / "device.txt").write_text(compiled.get_kernel_source())
