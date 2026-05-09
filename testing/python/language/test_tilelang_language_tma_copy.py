"""Test T.tma_copy() with user-managed synchronization.

For TMA loads (global -> shared):
  T.tma_copy() emits only expect_tx + tma_load (no arrive, no wait).
  The user must explicitly call T.barrier_arrive() and T.mbarrier_wait_parity().
  This allows multiple tma_copy operations to share a single barrier arrive.
  Pipeline buffer versioning expands the barrier to num_stages versions automatically.

For TMA stores (shared -> global):
  T.tma_copy() emits tma_store + tma_store_arrive (no wait).
  The user must explicitly call T.tma_store_wait() for synchronization.
  No barrier argument is needed for stores.
"""

from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T
import tilelang


def matmul_tma_copy(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
    num_stages,
):
    A_shape = (M, K)
    B_shape = (K, N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            mbar_A = T.alloc_barrier(128)
            mbar_B = T.alloc_barrier(128)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.tma_copy(A[by * block_M, k * block_K], A_shared, barrier=mbar_A)
                T.barrier_arrive(mbar_A)
                T.tma_copy(B[k * block_K, bx * block_N], B_shared, barrier=mbar_B)
                T.barrier_arrive(mbar_B)
                T.mbarrier_wait_parity(mbar_A, k % 2)
                T.mbarrier_wait_parity(mbar_B, k % 2)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_tma_copy(num_stages):
    M, N, K = 1024, 1024, 1024
    block_M, block_N, block_K = 128, 128, 32
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    threads = 128

    program = matmul_tma_copy(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        threads,
        num_stages,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        return C.to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_copy_pipeline_2_stages():
    run_gemm_tma_copy(num_stages=2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_copy_pipeline_3_stages():
    run_gemm_tma_copy(num_stages=3)


def matmul_tma_copy_store(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
    num_stages,
):
    """GEMM using T.tma_copy for both load (global->shared) and store (shared->global)."""
    A_shape = (M, K)
    B_shape = (K, N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            mbar_A = T.alloc_barrier(128)
            mbar_B = T.alloc_barrier(128)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.tma_copy(A[by * block_M, k * block_K], A_shared, barrier=mbar_A)
                T.barrier_arrive(mbar_A)
                T.tma_copy(B[k * block_K, bx * block_N], B_shared, barrier=mbar_B)
                T.barrier_arrive(mbar_B)
                T.mbarrier_wait_parity(mbar_A, k % 2)
                T.mbarrier_wait_parity(mbar_B, k % 2)
                T.gemm(A_shared, B_shared, C_local)
            # Store result: fragment -> shared -> global via T.tma_copy (no barrier needed)
            T.copy(C_local, C_shared)
            T.tma_copy(C_shared, C[by * block_M, bx * block_N])
            T.tma_store_wait()

    return main


def run_gemm_tma_copy_store(num_stages):
    M, N, K = 1024, 1024, 1024
    block_M, block_N, block_K = 128, 128, 32
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    threads = 128

    program = matmul_tma_copy_store(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        threads,
        num_stages,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        return C.to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def fp4_tma_copy_roundtrip(M=128, N=256, block_M=64, block_N=128):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float4_e2m1fn),
        B: T.Tensor((M, N), T.float4_e2m1fn),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), T.float4_e2m1fn)
            mbar = T.alloc_barrier(128)
            T.tma_copy(A[by * block_M, bx * block_N], A_shared, barrier=mbar)
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.tma_copy(A_shared, B[by * block_M, bx * block_N])
            T.tma_store_wait()

    return main


def run_fp4_tma_copy_roundtrip():
    import re
    import torch

    M, N = 128, 256
    program = fp4_tma_copy_roundtrip(M=M, N=N)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    device_source = kernel.get_kernel_source()
    host_source = kernel.get_host_source()
    assert "CUtensorMap" in device_source
    assert "tl::tma_load" in device_source
    assert "tl::tma_store" in device_source
    assert host_source.count("__tvm_tensormap_create_tiled") >= 2

    def descriptor_init_block(desc_name):
        marker = f"[0].v_ptr) = {desc_name};"
        start = host_source.find(marker)
        assert start >= 0, f"Missing {desc_name} TensorMap initialization"
        end = host_source.find("TVMFFIFunctionCall(__tvm_tensormap_create_tiled_packed", start)
        assert end >= 0, f"Missing {desc_name} TensorMap creation call"
        return host_source[start:end]

    def stack_int(block, index):
        match = re.search(rf"\[{index}\]\.v_int64\)\s*=\s*\(\(int64_t\)(-?\d+)\);", block)
        assert match, f"Missing stack[{index}] integer assignment in:\n{block}"
        return int(match.group(1))

    # create_tma_descriptor(data_type, rank, global_addr,
    #   global_shape..., global_stride..., smem_box..., smem_stride...,
    #   interleave, swizzle, l2_promotion, oob_fill)
    expected_tma_args = {
        1: 13,  # CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B
        2: 2,  # rank
        4: 256,  # global_shape[0], reversed innermost dimension
        5: 128,  # global_shape[1]
        6: 1,  # raw innermost stride, ignored by CUDA encode
        7: 128,  # next global stride in bytes: 256 fp4 elements == 128 bytes
        8: 128,  # smem_box[0]: 128 fp4 elements == 64 bytes
        9: 64,  # smem_box[1]
        10: 1,  # element stride[0]
        11: 1,  # element stride[1]
        12: 0,  # CU_TENSOR_MAP_INTERLEAVE_NONE
        13: 2,  # CU_TENSOR_MAP_SWIZZLE_64B
        14: 2,  # CU_TENSOR_MAP_L2_PROMOTION_L2_128B
        15: 0,  # CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    }
    for desc_name in ("A_desc", "B_desc"):
        block = descriptor_init_block(desc_name)
        for index, expected in expected_tma_args.items():
            assert stack_int(block, index) == expected

    a = torch.randint(-128, 128, (M, N // 2), device="cuda", dtype=torch.int8)
    b = kernel(a)
    assert torch.equal(b.view(torch.int8), a)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(10, 0)
def test_fp4_tma_copy_roundtrip():
    run_fp4_tma_copy_roundtrip()


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_copy_store_pipeline_2_stages():
    run_gemm_tma_copy_store(num_stages=2)


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_copy_store_pipeline_3_stages():
    run_gemm_tma_copy_store(num_stages=3)


if __name__ == "__main__":
    tilelang.testing.main()
    # test_tma_copy_pipeline_2_stages()
