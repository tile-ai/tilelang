import torch
import tilelang.testing
import tilelang.language as T


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def matmul_nt(M, N, K, bM, bN, bK, in_dtype, out_dtype, accum_dtype):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=256) as (bx, by):
            A_shared = T.alloc_shared((bM, bK), in_dtype)
            B_shared = T.alloc_shared((bN, bK), in_dtype)
            C_tmem = T.alloc_tmem((bM, bN), accum_dtype)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((bM, bN), accum_dtype)
            C_shared = T.alloc_shared((bM, bN), out_dtype)

            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=2):
                T.copy(A[by * bM, k * bK], A_shared)
                T.copy(B[bx * bN, k * bK], B_shared)
                T.gemm(A_shared, B_shared, C_tmem, transpose_B=True, mbar=mbar, wg_wait=-1, clear_accum=k == 0)
                T.mbarrier_wait_parity(mbar, k % 2)

            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[by * bM, bx * bN])

    return main


def assert_matmul_correctness(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype):
    func = matmul_nt(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype)
    kernel = tilelang.compile(
        func,
        out_idx=-1,
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    assert out_dtype in [T.int32], "Currently only int32 is supported"
    assert accum_dtype in [T.int32], "Currently only int32 is supported"

    if in_dtype is T.int8:
        A = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
        B = torch.randint(-128, 128, (N, K), device="cuda", dtype=torch.int8)
    elif in_dtype is T.uint8:
        A = torch.randint(0, 256, (M, K), device="cuda", dtype=torch.uint8)
        B = torch.randint(0, 256, (N, K), device="cuda", dtype=torch.uint8)
    else:
        raise ValueError(f"Unsupported input dtype: {in_dtype}")

    C = kernel(A, B)

    ref_c = (A.float() @ B.T.float()).to(torch.int32)
    print(C)
    print(ref_c)
    diff = calc_diff(C, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(10)
def test_assert_matmul():
    assert_matmul_correctness(1024, 1024, 1024, 128, 128, 128, T.int8, T.int32, T.int32)
    assert_matmul_correctness(1024, 1024, 1024, 128, 128, 128, T.uint8, T.int32, T.int32)


if __name__ == "__main__":
    test_assert_matmul()
