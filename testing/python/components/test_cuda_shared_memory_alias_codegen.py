import tilelang
import tilelang.language as T
import tilelang.testing


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


if __name__ == "__main__":
    tilelang.testing.main()
