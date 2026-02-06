import torch
import tilelang
import tilelang.language as T
from tilelang.utils.tensor import torch_assert_close
import itertools


def ref_program(A, B):
    return (A.half() @ B.half().T).to(dtype=torch.float32)


def manual_check_prog(C, C_ref):
    torch_assert_close(C[0], C_ref[0], rtol=0.01, atol=0.1)


def supply_prog(args):
    a_param, b_param = args
    M, K = a_param.shape
    N, _ = b_param.shape
    a = (torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.01).to(dtype=torch.float8_e4m3fnuz)
    b = (torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.01).to(dtype=torch.float8_e4m3fnuz)
    return [a, b]


def get_configs():
    block_Ms = [32, 64, 128]
    block_Ns = [32, 64, 128]
    block_Ks = [64, 128]
    num_stages = [0]
    num_threads = [256]
    k_packs = [1, 2]
    gemm_types = ["ss", "rs"]

    valid_configs = []

    for m, n, k, stages, t, kp, gemm_type in itertools.product(block_Ms, block_Ns, block_Ks, num_stages, num_threads, k_packs, gemm_types):
        valid_configs.append(
            {
                "block_M": m,
                "block_N": n,
                "block_K": k,
                "num_stages": stages,
                "num_threads": t,
                "k_pack": kp,
                "gemm_type": gemm_type,
            }
        )
    return valid_configs


@tilelang.autotune(
    configs=get_configs(), cache_input_tensors=True, ref_prog=ref_program, manual_check_prog=manual_check_prog, supply_prog=supply_prog
)
@tilelang.jit(out_idx=[-1])
def fp8_matmul(M, N, K, block_M, block_N, block_K, num_stages, num_threads, k_pack, gemm_type):
    dtype = T.float8_e4m3fnuz
    accum_dtype = T.float32

    @T.prim_func
    def gemm_fp8_rs(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_local = T.alloc_fragment((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_local)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_local, B_shared, C_local, transpose_B=True, k_pack=k_pack, policy=T.GemmWarpPolicy.FullRow)

            T.copy(C_local, C[by * block_M, bx * block_N])

    @T.prim_func
    def gemm_fp8_ss(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True, k_pack=k_pack, policy=T.GemmWarpPolicy.FullRow)

            T.copy(C_local, C[by * block_M, bx * block_N])

    if gemm_type == "ss":
        return gemm_fp8_ss
    elif gemm_type == "rs":
        return gemm_fp8_rs
    else:
        raise ValueError(f"Invalid gemm_type: {gemm_type}")


def test_gemm_fp8(M, N, K):
    kernel = fp8_matmul(M, N, K)
    a = (torch.randn(M, K, dtype=torch.float16, device="cuda") * 0.01).to(dtype=torch.float8_e4m3fnuz)
    b = (torch.randn(N, K, dtype=torch.float16, device="cuda") * 0.01).to(dtype=torch.float8_e4m3fnuz)
    c = kernel(a, b)
    ref_c = ref_program(a, b)
    torch_assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("passed~")


if __name__ == "__main__":
    test_gemm_fp8(512, 512, 512)
