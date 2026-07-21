import pytest
import tilelang
import tilelang.language as T
import tilelang.testing


def _get_dynamic_shared_memory_bytes(artifact):
    device_funcs = list(artifact.device_mod.functions.values())
    assert len(device_funcs) == 1
    return int(device_funcs[0].attrs["dyn_shared_memory_buf"])


@tilelang.testing.requires_cuda
def test_dynamic_shared_memory_merge_emits_named_aliases():
    @T.prim_func
    def kernel(
        A: T.Tensor((32,), T.float16),
        B: T.Tensor((32,), T.float16),
        C: T.Tensor((32,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            A_shared = T.alloc_shared((32,), T.float16)
            B_shared = T.alloc_shared((32,), T.float16)
            A_shared[0] = A[0]
            B_shared[0] = B[0]
            T.tvm_storage_sync("shared")
            C[0] = A_shared[0] + B_shared[0]

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source

    assert "extern __shared__ __align__(1024) uchar buf_dyn_shmem[];" in source
    assert "void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));" in source
    assert "void* B_shared = ((void*)((char*)buf_dyn_shmem + 64));" in source
    assert "A_shared" in source
    assert "B_shared" in source
    assert "((half_t*)buf_dyn_shmem)" not in source
    assert _get_dynamic_shared_memory_bytes(artifact) == 128


@tilelang.testing.requires_cuda
def test_static_and_local_fp4_allocations_use_packed_extents():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), T.float4_e2m1fn, scope="shared")
            A_local = T.alloc_local((127,), T.float4_e2m1fn)
            A_shared[126] = T.cast(0.0, T.float4_e2m1fn)
            A_local[126] = A_shared[126]

    source = tilelang.lower(kernel, target="cuda").kernel_source

    assert "fp4_e2_t A_shared[64];" in source
    assert "fp4_e2_2_t A_local_packed[64];" in source


@tilelang.testing.requires_cuda
def test_single_dynamic_fp4_allocation_uses_packed_size():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), T.float4_e2m1fn)
            A_shared[126] = T.cast(0.0, T.float4_e2m1fn)

    artifact = tilelang.lower(kernel, target="cuda")

    assert "extern __shared__ __align__(1024) fp4_e2_t A_shared[];" in artifact.kernel_source
    assert _get_dynamic_shared_memory_bytes(artifact) == 64


@tilelang.testing.requires_cuda
def test_merged_dynamic_fp4_allocations_use_packed_sizes_and_offsets():
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), T.float4_e2m1fn)
            B_shared = T.alloc_shared((127,), T.float4_e2m1fn)
            A_shared[126] = T.cast(0.0, T.float4_e2m1fn)
            B_shared[126] = T.cast(1.0, T.float4_e2m1fn)
            T.tvm_storage_sync("shared")
            A_shared[0] = B_shared[126]

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source

    assert "extern __shared__ __align__(1024) uchar buf_dyn_shmem[];" in source
    assert "void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));" in source
    assert "void* B_shared = ((void*)((char*)buf_dyn_shmem + 64));" in source
    assert _get_dynamic_shared_memory_bytes(artifact) == 128


@tilelang.testing.requires_cuda
@pytest.mark.parametrize(
    "dtype,storage_type",
    [(T.int4, "signed char"), (T.dtype("uint4"), "uchar")],
)
def test_static_packed_int4_allocation_uses_packed_extent(dtype, storage_type):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), dtype, scope="shared")
            A_shared[126] = T.cast(1, dtype)

    source = tilelang.lower(kernel, target="cuda").kernel_source

    assert f"{storage_type} A_shared[64];" in source


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.int4, T.dtype("uint4")])
def test_single_dynamic_packed_int4_allocation_uses_packed_size(dtype):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), dtype)
            A_shared[126] = T.cast(1, dtype)

    artifact = tilelang.lower(kernel, target="cuda")

    assert _get_dynamic_shared_memory_bytes(artifact) == 64


@tilelang.testing.requires_cuda
@pytest.mark.parametrize("dtype", [T.int4, T.dtype("uint4")])
def test_merged_dynamic_packed_int4_allocations_use_packed_sizes_and_offsets(dtype):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), dtype)
            B_shared = T.alloc_shared((127,), dtype)
            A_shared[126] = T.cast(0, dtype)
            B_shared[126] = T.cast(1, dtype)
            T.tvm_storage_sync("shared")
            A_shared[0] = B_shared[126]

    artifact = tilelang.lower(kernel, target="cuda")
    source = artifact.kernel_source

    assert "void* A_shared = ((void*)((char*)buf_dyn_shmem + 0));" in source
    assert "void* B_shared = ((void*)((char*)buf_dyn_shmem + 64));" in source
    assert _get_dynamic_shared_memory_bytes(artifact) == 128


if __name__ == "__main__":
    tilelang.testing.main()
