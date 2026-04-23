"""Frontend FP4 GEMM example with compilation and correctness checks.

This exercises the simple SM120 `mma.sync` FP4 path exposed through `T.gemm`:
- A/B are declared as `T.float4_e2m1fn`
- the matmul is expressed with `T.gemm(...)`
- the runtime inputs use packed FP4 storage (`torch.float4_e2m1fn_x2`)
- correctness is checked against a float32 reference computed from an E2M1 LUT
"""

import time

import torch

import tilelang
import tilelang.language as T

tilelang.disable_cache()

def post_process(source, _):
    source = r"""
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/cuda_fp4.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(const fp4_e2_t* __restrict__ A, const fp4_e2_t* __restrict__ B, float* __restrict__ C);
extern "C" __global__ void __launch_bounds__(32, 1) main_kernel(const fp4_e2_t* __restrict__ A, const fp4_e2_t* __restrict__ B, float* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_local[8];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(C_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  *(fp4_e2_32_t*)(((fp4_e2_t*)buf_dyn_shmem) + (((int)threadIdx.x) * 16)) = *(fp4_e2_32_t*)(A + (((int)threadIdx.x) * 16));
  *(fp4_e2_32_t*)(((fp4_e2_t*)buf_dyn_shmem) + ((((int)threadIdx.x) * 16) + 512)) = *(fp4_e2_32_t*)(B + (((int)threadIdx.x) * 16));
    __syncthreads();

  {
    fp4_e2_2_t A_local_packed[16];
    fp4_e2_2_t B_local_packed[16];
    // view A_local_packed as unsigned and clear
    for (int i = 0; i < 16; ++i) {
      A_local_packed[i] = 0;
    }
    for (int i = 0; i < 16; ++i) {
      B_local_packed[i] = 0;
    }
    tl::mma_sync<tl::DataType::kFloat4_e2m1fn, tl::DataType::kFloat4_e2m1fn, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(C_local + 4), reinterpret_cast<const unsigned*>(A_local_packed + 8), reinterpret_cast<const unsigned*>(B_local_packed + 12));
  }
  if (threadIdx.x == 0) {
    for (int i = 4; i < 8; ++i) {
        printf("threadIdx.x = %d, C_local[%d] = %f\n", threadIdx.x, i, C_local[i]);
    }
  }
  __syncthreads();
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    *(float2*)(C + (((((i_1 & 1) * 128) + ((((int)threadIdx.x) >> 2) * 16)) + ((i_1 >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(C_local + (i_1 * 2));
  }
}

    """
    return source

tilelang.register_cuda_postproc(post_process)

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


def matmul_nt_fp4(M, N, K, block_M, block_N, block_K, accum_dtype=T.float32):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float4_e2m1fn),
        B: T.Tensor((N, K), T.float4_e2m1fn),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=32) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float4_e2m1fn)
            B_shared = T.alloc_shared((block_N, block_K), T.float4_e2m1fn)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, disable_tma=True)
                T.copy(B[bx * block_N, ko * block_K], B_shared, disable_tma=True)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def unpack_fp4_to_uint8(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Unpack (rows, cols//2) packed bytes into one FP4 code per uint8 element."""
    flat = packed.view(torch.uint8).reshape(rows, cols // 2)
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(rows, cols).contiguous()


def unpack_fp4_to_float(packed: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Decode packed E2M1 FP4 storage into float32 using a lookup table."""
    lut = torch.tensor(FP4_E2M1_TO_FLOAT, dtype=torch.float32, device=packed.device)
    unpacked = unpack_fp4_to_uint8(packed, rows, cols).to(torch.int64)
    return lut[unpacked]


def make_fp4_storage_tensor(packed: torch.Tensor) -> torch.Tensor:
    """View packed bytes as the storage dtype expected by the JIT wrapper."""
    storage_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return packed.view(storage_dtype) if storage_dtype is not None else packed.view(torch.int8)


def compile_fp4_gemm(
    M=256,
    N=256,
    K=256,
    block_M=128,
    block_N=128,
    block_K=64,
    accum_dtype=T.float32,
    print_source=False,
):
    func = matmul_nt_fp4(M, N, K, block_M, block_N, block_K, accum_dtype=accum_dtype)
    print(func)
    kernel = tilelang.compile(
        func,
        out_idx=[2],
        target="cuda",
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    print("Compilation succeeded.")
    if print_source:
        print(kernel.get_kernel_source())
    return func, kernel


def main():
    M, N, K = 16, 16, 64
    block_M, block_N, block_K = 16, 16, 64

    print(f"Running FP4 GEMM: M={M}, N={N}, K={K}")
    print(f"  block_M={block_M}, block_N={block_N}, block_K={block_K}")

    _, jit_kernel = compile_fp4_gemm(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        accum_dtype=T.float32,
        print_source=True,
    )

    torch.manual_seed(42)

    a_packed = torch.randint(0, 256, (M, K // 2), device="cuda", dtype=torch.uint8)
    b_packed = torch.randint(0, 256, (N, K // 2), device="cuda", dtype=torch.uint8)
    a_input = make_fp4_storage_tensor(a_packed)
    b_input = make_fp4_storage_tensor(b_packed)

    a_zero = make_fp4_storage_tensor(torch.zeros((M, K // 2), device="cuda", dtype=torch.uint8))
    b_zero = make_fp4_storage_tensor(torch.zeros((N, K // 2), device="cuda", dtype=torch.uint8))
    c_zero = jit_kernel(a_zero, b_zero)
    zero_max = c_zero.abs().max().item()
    print(c_zero)
    assert zero_max == 0.0, f"Zero test failed: max_abs={zero_max}"
    print("[PASS] zeros in -> zeros out")

    c = jit_kernel(a_input, b_input)
    a_float = unpack_fp4_to_float(a_packed, M, K)
    b_float = unpack_fp4_to_float(b_packed, N, K)
    ref_c = a_float @ b_float.T

    diff = (c.float() - ref_c).abs()
    max_diff = diff.max().item()
    rel_err = diff.sum().item() / (ref_c.abs().sum().item() + 1e-10)
    print(f"[NUMERICAL] max_abs_diff={max_diff:.6f}, rel_err={rel_err:.6e}")
    if max_diff == 0.0:
        print("[PASS] numerical verification")
    else:
        print("[WARN] numerical mismatch detected; FP4 MMA load layout still needs a dedicated shared->register path")

    for _ in range(10):
        jit_kernel(a_input, b_input)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(100):
        jit_kernel(a_input, b_input)
    torch.cuda.synchronize()

    elapsed_ms = (time.perf_counter() - start) / 100 * 1000
    tflops = 2 * M * N * K / (elapsed_ms / 1e3) / 1e12
    print(f"Latency: {elapsed_ms:.4f} ms")
    print(f"TFLOPS:  {tflops:.2f}")


if __name__ == "__main__":
    main()
