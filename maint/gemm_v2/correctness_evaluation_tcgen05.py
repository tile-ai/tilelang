# pytest correctness_evaluation.py -n 32
import pytest
from tilelang import tvm as tvm
import tilelang.testing
import tilelang.language as T


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_tmem = T.alloc_tmem([block_M, block_N], accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_tmem, trans_A, trans_B, mbar=mbar, wg_wait=-1, clear_accum=k == 0)
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)

            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def _compile_and_check(
    program,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
):
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    print(kernel.get_kernel_source())

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if in_dtype == T.float32:
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)
    print("assert_allclose")


def run_gemm(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=2,
    num_threads=128,
):
    if block_N >= 256 or block_M >= 256 or block_K >= 256:
        num_stages = 0
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    _compile_and_check(program, trans_A, trans_B, in_dtype, out_dtype)


M_VALUES = [32, 64, 128, 256]
N_VALUES = [64, 128, 256, 512]
K_VALUES = [16, 32, 64, 128]
K_VALUES_8Bit = [32, 64, 128]
FALSE_TRUE_CASES = [
    pytest.param(
        k,
        T.float16,
        T.float32,
        T.float32,
        id=f"K{k}-float16-float-float",
    )
    for k in K_VALUES
] + [
    pytest.param(
        k,
        T.float8_e5m2,
        T.float32,
        T.float32,
        id="K32-float8_e5m2-float32-float32",
    )
    for k in K_VALUES_8Bit
]

TRANS_CASES = [
    pytest.param(False, True, id="nt"),
]


@pytest.mark.parametrize("m", M_VALUES, ids=lambda v: f"M{v}")
@pytest.mark.parametrize("n", N_VALUES, ids=lambda v: f"N{v}")
@pytest.mark.parametrize("k,in_dtype,out_dtype,accum_dtype", FALSE_TRUE_CASES)
def test_gemm_false_true(m, n, k, in_dtype, out_dtype, accum_dtype):
    import torch

    required_torch_attrs = {
        in_dtype,
        out_dtype,
        accum_dtype,
    }
    for attr in required_torch_attrs:
        if not hasattr(torch, attr):
            pytest.skip(f"Torch does not expose dtype {attr}")
    run_gemm(
        m,
        n,
        k * 3,
        False,
        True,
        in_dtype,
        out_dtype,
        accum_dtype,
        m,
        n,
        k,
    )


if __name__ == "__main__":
    tilelang.testing.main()

    # # Test Pass
    # for m in [32, 64, 128, 256]:
    #     for n in [16, 32, 64, 128]:
    #         for k in [16, 32, 64, 128]:
    #             if m in [32, 64] and (n not in [64, 128, 256]):
    #                 continue
    #             print(f"======================= Test {m} {n} {k} False True =============================")
    #             run_gemm(m, n, k * 3, False, True, T.float16, T.float, T.float, m, n, k, 2, 128)
    #             print(f"Test {m} {n} {k} Pass")

    # # Test Pass
    # for m in [32, 64, 128, 256]:
    #     for n in [32, 64, 128]:
    #         for k in [16, 32, 64, 128]:
    #             if m in [32, 64] and (n not in [64, 128, 256]):
    #                 continue
    #             print(f"======================= Test {m} {n} {k} False True =============================")
    #             run_gemm(m, n, k * 3, False, True, T.float16, T.float, T.float, m, n, k, 2, 256)
    #             print(f"Test {m} {n} {k} Pass")

    # # Test Pass
    # for m in [32, 64, 128, 256]:
    #     for n in [16, 32, 64, 128]:
    #         for k in [32, 64, 128]:
    #             if m in [32, 64] and (n not in [64, 128, 256]):
    #                 continue
    #             print(f"======================= Test {m} {n} {k} False True =============================")
    #             run_gemm(m, n, k * 3, False, True, T.float8_e5m2, T.float, T.float, m, n, k, 2, 128)
