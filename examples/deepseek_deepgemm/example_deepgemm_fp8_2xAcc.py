from typing import Tuple

import torch
import tilelang.testing
import tilelang
import tilelang.language as T
from tilelang.utils.tensor import map_torch_type

tilelang.testing.set_random_seed(42)


@tilelang.jit
def tl_gemm(
    M,
    N,
    K,
    block_N,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        T.float8_e4m3fn,
    ], "Currently only float8_e4m3 is supported"
    assert out_dtype in [
        T.bfloat16,
        T.float32,
    ], "Currently only float16 and float32 are supported"

    group_size = 128
    block_M = 128
    block_K = 128

    A_shape = (M, K)
    Scales_A_shape = (M, T.ceildiv(K, group_size))
    B_shape = (N, K)
    Scales_B_shape = (T.ceildiv(N, group_size), T.ceildiv(K, group_size))
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (block_M, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
        scales_a: T.Tensor(Scales_A_shape, T.float32),
        scales_b: T.Tensor(Scales_B_shape, T.float32),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_shared = T.alloc_shared(C_shared_shape, out_dtype)
            Scale_C_shared = T.alloc_shared((block_M), T.float32)
            C_local = T.alloc_fragment(C_shared_shape, accum_dtype)
            C_local_accum = T.alloc_fragment(C_shared_shape, accum_dtype)

            # Improve L2 Cache
            T.use_swizzle(panel_size=10)

            T.clear(C_local)
            T.clear(C_local_accum)
            K_iters = T.ceildiv(K, block_K)
            for k in T.Pipelined(K_iters, num_stages=4):
                # Load A into shared memory
                T.copy(A[by * block_M, k * block_K], A_shared)
                # Load B into shared memory
                T.copy(B[bx * block_N, k * block_K], B_shared)
                # Load scale into shared memory
                Scale_B = scales_b[bx * block_N // group_size, k]
                for i in T.Parallel(block_M):
                    Scale_C_shared[i] = scales_a[by * block_M + i, k] * Scale_B

                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                # Promote to enable 2xAcc
                for i, j in T.Parallel(block_M, block_N):
                    C_local_accum[i, j] += C_local[i, j] * Scale_C_shared[i]
                T.clear(C_local)
            # TMA store
            T.copy(C_local_accum, C_shared)
            T.copy(C_shared, C[by * block_M, bx * block_N])

    return main


def ceildiv(a, b):
    return (a + b - 1) // b


def per_token_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2 and x.size(1) % 128 == 0
    m, n = x.shape
    x_view = x.view(m, -1, 128)
    x_amax = x_view.abs().float().amax(dim=2).view(m, -1).clamp(1e-4)
    return (x_view * (448.0 / x_amax.unsqueeze(2))).to(torch.float8_e4m3fn).view(m, n), (x_amax / 448.0).view(m, -1)


def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(ceildiv(m, 128) * 128, ceildiv(n, 128) * 128, dtype=x.dtype, device=x.device)
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    x_scaled = (x_view * (448.0 / x_amax)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), (x_amax / 448.0).view(x_view.size(0), x_view.size(2))


def ref_deepgemm_fp8(A_fp8, B_fp8, A_scale, B_scale, out_dtype):
    # A_scale: (M, K//128)       ==>   (M//128, K//128, 128)
    # B_scale: (N//128, K//128)  ==>   (N//128, K//128, 128)
    # A_fp8: (M, K)
    # B_fp8: (N, K)
    # out_dtype: float16 or float32
    # return C: (M, N)
    M, N, K = A_fp8.shape[0], B_fp8.shape[0], A_fp8.shape[1]
    A_scales = A_scale.view(M // 128, 128, K // 128).permute(0, 2, 1)
    B_scales = B_scale.repeat_interleave(128, dim=1).view(N // 128, K // 128, 128)
    C = torch.zeros(M, N, device="cuda", dtype=out_dtype)
    c_acc = torch.zeros(128, 128, device="cuda", dtype=torch.float32)
    for i in range(ceildiv(M, 128)):
        for j in range(ceildiv(N, 128)):
            c_acc.zero_()
            for k in range(ceildiv(K, 128)):
                c = torch._scaled_mm(
                    A_fp8[i * 128 : (i + 1) * 128, k * 128 : (k + 1) * 128],
                    B_fp8[j * 128 : (j + 1) * 128, k * 128 : (k + 1) * 128].T,
                    scale_a=A_scales[i, k].view(128, 1).contiguous(),
                    scale_b=B_scales[j, k].view(1, 128).contiguous(),
                    out_dtype=torch.bfloat16,
                )
                c_acc += c.to(torch.float32)
            C[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = c_acc.to(out_dtype)
    return C


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def assert_tl_gemm_correctness(M, N, K, block_N, in_dtype, out_dtype, accum_dtype):
    kernel = tl_gemm(M, N, K, block_N, in_dtype, out_dtype, accum_dtype)
    src_code = kernel.get_kernel_source()

    # src_code is the generated cuda source
    assert src_code is not None

    in_dtype = map_torch_type(in_dtype)
    out_dtype = map_torch_type(out_dtype)
    accum_dtype = map_torch_type(accum_dtype)

    A = torch.randn(M, K).to(torch.bfloat16).cuda()
    B = torch.randn(N, K).to(torch.bfloat16).cuda()
    A_fp8, A_scale = per_token_cast_to_fp8(A.clone())
    B_fp8, B_scale = per_block_cast_to_fp8(B.clone())

    C = torch.zeros(M, N, device="cuda", dtype=out_dtype)

    kernel(A_fp8, B_fp8, C, A_scale, B_scale)
    # Get Reference Result
    ref_c = ref_deepgemm_fp8(A_fp8, B_fp8, A_scale, B_scale, out_dtype)
    diff = calc_diff(C, ref_c)
    print(f"diff: {diff}")
    assert diff < 1e-3

    profiler = kernel.get_profiler()
    latency = profiler.do_bench(warmup=25)
    # Ensure that the latency is not None
    assert latency is not None
    print(f"latency: {latency} ms")
    tflops = 2 * M * N * K / latency / 1e9
    print(f"tflops: {tflops}")


def main():
    assert_tl_gemm_correctness(1024, 1024, 8192, 128, T.float8_e4m3fn, T.bfloat16, T.float32)


if __name__ == "__main__":
    for dtype in [T.float8_e4m3fn]:
        for out_dtype in [T.bfloat16, T.float32]:
            for block_N in [16, 32, 64, 128]:
                assert_tl_gemm_correctness(1024, 1024, 8192, block_N, dtype, out_dtype, T.float32)
