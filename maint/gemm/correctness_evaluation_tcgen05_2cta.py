# pytest correctness_evaluation_tcgen05_2cta.py -n 32
import pytest
from tilelang import tvm as tvm
import tilelang
import tilelang.testing
import tilelang.language as T

tilelang.disable_cache()


def matmul_2cta(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(N, block_N), threads=128, cluster_dims=2) as (bx, by):
            A_shared = T.alloc_shared((num_stages, block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((num_stages, block_K, block_N // 2), in_dtype)  # each CTA holds half of B
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            loaded = T.alloc_cluster_barrier([32 * 2] * num_stages)
            consumed = T.alloc_cluster_barrier([1] * num_stages)
            tmem_full = T.alloc_barrier([1])

            tx = T.get_thread_binding()
            cta_id = T.block_rank_in_cluster()
            T.assume(cta_id < 2)

            T.use_swizzle(16)

            if tx < 32:  # warp 0: issue TMA loads
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.mbarrier_wait_parity(consumed[k % num_stages], ((k // num_stages) & 1) ^ 1)
                    T.tma_copy(
                        A[bx * block_M : (bx + 1) * block_M, k * block_K : (k + 1) * block_K],
                        A_shared[k % num_stages, :, :],
                        barrier=loaded[k % num_stages],
                    )
                    T.tma_copy(
                        B[k * block_K : (k + 1) * block_K, (by * 2 + cta_id) * (block_N // 2) : (by * 2 + cta_id + 1) * (block_N // 2)],
                        B_shared[k % num_stages, :, :],
                        barrier=loaded[k % num_stages],
                    )
                    T.mbarrier_arrive(loaded[k % num_stages], 0)  # arrive on leader CTA's barrier
            elif cta_id == 0 and tx < 64:  # warp 1 on leader CTA: issue tcgen5 MMA
                for k in T.serial(T.ceildiv(K, block_K)):
                    T.mbarrier_wait_parity(loaded[k % num_stages], (k // num_stages) & 1)
                    T.tcgen05_gemm(
                        A_shared[k % num_stages, :, :],
                        B_shared[k % num_stages, :, :],
                        C_tmem,
                        mbar=consumed[k % num_stages],
                        clear_accum=k == 0,
                        use_2cta=True,
                    )
                T.tcgen05_mma_arrive(tmem_full, arrive_2cta=True)

            T.mbarrier_wait_parity(tmem_full, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[bx * block_M, by * block_N])

    return main


def _compile_and_check(program, out_dtype):
    kernel = tilelang.compile(program, out_idx=[2])

    print(kernel.get_kernel_source())

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    def ref_program(A, B):
        import torch

        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        return C.to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
    print("assert_allclose passed")


def run_gemm(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
    block_M,
    block_N,
    block_K,
    num_stages=4,
):
    program = matmul_2cta(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        num_stages,
    )
    _compile_and_check(program, out_dtype)


M_VALUES = [64, 128, 256]
N_VALUES = [64, 128, 256]

# atom_k=16 for fp16/bf16 (K%16==0), atom_k=32 for fp8/int8 (K%32==0)
K_VALUES_16 = [16, 32, 64, 128]
K_VALUES_32 = [32, 64, 128]

# Dtype cases: (block_K, in_dtype, out_dtype, accum_dtype)
FP16_CASES = [pytest.param(k, T.float16, T.float32, T.float32, id=f"K{k}-fp16-fp32-fp32") for k in K_VALUES_16]

FP8_E5M2_CASES = [pytest.param(k, T.float8_e5m2, T.float32, T.float32, id=f"K{k}-fp8e5m2-fp32-fp32") for k in K_VALUES_32]

INT8_CASES = [pytest.param(k, T.int8, T.int32, T.int32, id=f"K{k}-int8-int32-int32") for k in K_VALUES_32]

ALL_DTYPE_CASES = FP16_CASES + FP8_E5M2_CASES + INT8_CASES


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("block_k,in_dtype,out_dtype,accum_dtype", ALL_DTYPE_CASES)
def test_gemm_2cta(m, n, block_k, in_dtype, out_dtype, accum_dtype):
    import torch

    for attr in {in_dtype, out_dtype, accum_dtype}:
        if not hasattr(torch, attr):
            pytest.skip(f"Torch does not expose dtype {attr}")

    # M = 2 * block_M so ceildiv(M, block_M) = 2 (cluster needs >= 2 tiles in M dim)
    # K = 3 * block_K to exercise multi-iteration pipelining
    k = block_k * 3
    run_gemm(
        m * 2,
        n,
        k,
        in_dtype,
        out_dtype,
        accum_dtype,
        block_M=m,
        block_N=n,
        block_K=block_k,
    )


if __name__ == "__main__":
    tilelang.testing.main()
