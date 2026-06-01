import torch
import tilelang
import tilelang.language as T


FP4_E2M1_TO_FLOAT = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def fp4_uint8_to_float(tensor):
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=tensor.device)
    return lut[(tensor.to(torch.uint8) & 0x0F).to(torch.int64)]


@tilelang.jit(
    target="cuda",
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def matmul(A, B, block_M, block_N, block_K, out_dtype=T.float32, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), T.float4_e2m1fn)
    B: T.Tensor((N, K), T.float4_e2m1fn)
    C = T.empty((M, N), out_dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float4_e2m1fn)
        B_shared = T.alloc_shared((block_N, block_K), T.float4_e2m1fn)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def test_gemm_fp4(M, N, K):
    a = torch.randint(0, 16, (M, K), dtype=torch.uint8, device="cuda")
    b = torch.randint(0, 16, (N, K), dtype=torch.uint8, device="cuda")

    c = matmul(a, b, 128, 128, 128)
    ref_c = fp4_uint8_to_float(a) @ fp4_uint8_to_float(b).T

    diff = calc_diff(c, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3


def main():
    test_gemm_fp4(1024, 1024, 1024)


def run_regression_perf():
    M, N, K = 4096, 4096, 4096
    kernel = matmul.compile(M=M, N=N, K=K, block_M=128, block_N=128, block_K=128)
    profiler = kernel.get_profiler(tilelang.TensorSupplyType.Integer)
    if torch.version.hip is None:
        return profiler.do_bench(backend="cupti")
    return profiler.do_bench()


if __name__ == "__main__":
    main()
