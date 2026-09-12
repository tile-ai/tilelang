import tilelang
import tilelang.language as T
import tilelang.testing
import torch


@T.prim_func
def histogram(
    indices: T.Tensor((64,), "int32"),
    counts: T.Tensor((4,), "int32"),
    positions: T.Tensor((64,), "int32"),
):
    with T.Kernel(1, threads=64):
        lane = T.get_thread_binding(0)
        shared = T.alloc_shared((4,), "int32")
        if lane < 4:
            shared[lane] = 0
        T.sync_threads()
        bucket = indices[lane]
        positions[lane] = T.atomic_add(shared[bucket], 1, return_prev=True)
        T.sync_threads()
        if lane < 4:
            counts[lane] = shared[lane]


def test_int32_atomic_add_lowers_to_metal_atomic_fetch_add():
    artifact = tilelang.lower(
        histogram,
        target="metal",
        enable_host_codegen=False,
        enable_device_compile=False,
    )
    assert "atomic_fetch_add_explicit" in artifact.kernel_source
    assert "threadgroup atomic_int" in artifact.kernel_source


@tilelang.testing.requires_metal
def test_threadgroup_atomic_add_returns_previous_value():
    indices = torch.arange(64, dtype=torch.int32, device="mps") % 4
    counts = torch.empty(4, dtype=torch.int32, device="mps")
    positions = torch.empty(64, dtype=torch.int32, device="mps")
    compiled = tilelang.compile(
        histogram,
        out_idx=[],
        target="metal",
        target_host="c",
        execution_backend="tvm_ffi",
    )
    compiled(indices, counts, positions)
    torch.mps.synchronize()
    torch.testing.assert_close(counts.cpu(), torch.full((4,), 16, dtype=torch.int32))
    assert sorted(positions.cpu().tolist()) == sorted(list(range(16)) * 4)


if __name__ == "__main__":
    tilelang.testing.main()
