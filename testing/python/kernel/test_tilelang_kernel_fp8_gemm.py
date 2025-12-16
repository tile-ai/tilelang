import torch
import tilelang.testing
import tilelang.language as T
from tilelang.utils.tensor import map_torch_type


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
        with T.Kernel(T.ceildiv(N, bN), T.ceildiv(M, bM), threads=128) as (bx, by):
            A_shared = T.alloc_shared((bM, bK), in_dtype)
            B_shared = T.alloc_shared((bN, bK), in_dtype)
            C_local = T.alloc_fragment((bM, bN), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, bK), num_stages=3):
                T.copy(A[by * bM, k * bK], A_shared)
                T.copy(B[bx * bN, k * bK], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * bM, bx * bN])

    return main


def assert_matmul_correctness(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype):
    func = matmul_nt(M, N, K, block_M, block_N, block_K, in_dtype, out_dtype, accum_dtype)
    kernel = tilelang.compile(func, out_idx=-1)

    A = torch.randn(M, K).to(map_torch_type(in_dtype)).cuda()
    B = torch.randn(N, K).to(map_torch_type(in_dtype)).cuda()

    C = kernel(A, B)

    ref_c = torch.matmul(A.to(map_torch_type(accum_dtype)), B.T.to(map_torch_type(accum_dtype))).to(map_torch_type(out_dtype))
    print(C)
    print(ref_c)
    diff = calc_diff(C, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version(9)
def test_assert_matmul():
    assert_matmul_correctness(1024, 1024, 1024, 128, 128, 64, T.float8_e4m3fn, T.float32, T.float32)
    assert_matmul_correctness(1024, 1024, 1024, 128, 128, 64, T.float8_e5m2, T.float32, T.float32)


if __name__ == "__main__":
    tilelang.testing.main()
