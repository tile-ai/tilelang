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


def unpack_fp4_storage_to_float(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    packed_u8 = packed.to(torch.uint8).reshape(rows, cols // 2)
    lo = packed_u8 & 0x0F
    hi = (packed_u8 >> 4) & 0x0F
    values = torch.stack([lo, hi], dim=-1).reshape(rows, cols).to(torch.int64)
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed.device)
    return lut[values]


def require_sm120():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    major, _ = torch.cuda.get_device_capability()
    if major < 12:
        raise RuntimeError("SM120 A8W4 GEMM requires an SM120+ CUDA device")


def matmul_a8w4(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    out_dtype,
    accum_dtype,
    num_stages=2,
    threads=128,
):
    if K % 32 != 0 or block_K % 32 != 0 or block_K > K:
        raise ValueError("matmul_a8w4 requires K and block_K to be multiples of 32 and block_K <= K")

    A_shape = (M, K)
    B_shape = (N, K)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, T.float8_e4m3fn),
        B: T.Tensor(B_shape, T.float4_e2m1fn),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float8_e4m3fn)
            B_shared = T.alloc_shared((block_N, block_K), T.float4_e2m1fn)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def main():
    require_sm120()

    M, N, K = 256, 256, 256
    block_M, block_N, block_K = 128, 128, 64
    func = matmul_a8w4(M, N, K, block_M, block_N, block_K, T.float32, T.float32)
    kernel = tilelang.compile(
        func,
        out_idx=[2],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )

    torch.manual_seed(0)
    a_f16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    a = a_f16.to(torch.float8_e4m3fn)
    b = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8).view(torch.int8)

    c_zero = kernel(torch.zeros_like(a), torch.zeros_like(b))
    assert c_zero.abs().max().item() == 0.0

    c = kernel(a, b)
    ref = a.to(torch.float32) @ unpack_fp4_storage_to_float(b, N, K).T
    diff = (c.float() - ref).abs()
    rel_err = diff.sum().item() / (ref.abs().sum().item() + 1e-10)
    assert diff.max().item() <= 1e-3
    print(f"max_abs_diff={diff.max().item():.6f}, rel_err={rel_err:.6f}")


if __name__ == "__main__":
    main()
