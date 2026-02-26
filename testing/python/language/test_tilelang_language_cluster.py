import tilelang
import tilelang.language as T
import tilelang.testing
import torch


def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128, cluster_dims=(2, 1, 1)) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


@tilelang.jit(out_idx=-1)
def get_cta_rank_in_cluster(cluster_size=4):
    assert 128 % cluster_size == 0

    @T.prim_func
    def main(A: T.Tensor((128), T.int32)):
        with T.Kernel(128, cluster_dims=(cluster_size, 1, 1)) as bx:
            T.cluster_arrive()
            T.cluster_wait()
            cta_rank_in_cluster = T.block_rank_in_cluster()
            if T.get_thread_binding() == 0:
                A[bx] = cta_rank_in_cluster
            T.cluster_sync()

    return main


@tilelang.jit(out_idx=-1)
def barrier_kernel():
    @T.prim_func
    def main(A: T.Tensor((128), T.int32)):
        with T.Kernel(128, threads=128, cluster_dims=(4, 1, 1)):
            mbar = T.alloc_cluster_barrier([256])
            T.cluster_sync()
            T.mbarrier_arrive(mbar, 0)
            if T.block_rank_in_cluster() == 0:
                T.mbarrier_wait_parity(mbar, 0)
            T.cluster_sync()

    return main


def run_cython_cluster_launch():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)
    mod = tilelang.compile(kernel, execution_backend="cython")
    assert "clusterDim = {2, 1, 1}" in mod.get_host_source()


def run_tvm_ffi_cluster_launch():
    kernel = matmul(1024, 1024, 1024, 128, 128, 32)
    mod = tilelang.compile(kernel, execution_backend="tvm_ffi")
    check_str = r"""
  (((TVMFFIAny*)stack_ffi_any)[3].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[3].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[3].v_int64) = ((int64_t)2);
  (((TVMFFIAny*)stack_ffi_any)[4].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[4].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[4].v_int64) = ((int64_t)1);
  (((TVMFFIAny*)stack_ffi_any)[5].type_index) = 1;
  (((TVMFFIAny*)stack_ffi_any)[5].zero_padding) = 0;
  (((TVMFFIAny*)stack_ffi_any)[5].v_int64) = ((int64_t)1);
"""
    assert check_str in mod.get_host_source()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_cluster_launch():
    run_cython_cluster_launch()
    run_tvm_ffi_cluster_launch()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_cluster_launch_intrinsics(cluster_size=4):
    kernel = get_cta_rank_in_cluster(cluster_size)
    result = kernel()
    ref = torch.arange(128, dtype=torch.int32, device="cuda") % cluster_size
    assert torch.all(result == ref)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_cluster_barrier():
    kernel = barrier_kernel()
    kernel()


if __name__ == "__main__":
    tilelang.testing.main()
