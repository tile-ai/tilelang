import tilelang
import tilelang.language as T

import tvm


@tvm.register_func("tilelang_callback_cuda_postproc", override=True)
def tilelang_callback_cuda_postproc(code, _):
    code = """
#include <nvshmem.h>
#include <nvshmemx.h>
#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>

extern "C" __global__ void main_kernel(short* __restrict__ A, short* __restrict__ B);
extern "C" __global__ void __launch_bounds__(128) main_kernel(short* __restrict__ A, short* __restrict__ B) {
  int mype[1];
  extern __shared__ __align__(1024) short A_shared[];
  mype[0] = nvshmem_my_pe();
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("mype: %d\\n", mype[0]);
  }
}"""
    return code


def dist_test(M, N, block_M, block_N, dtype="int16"):
    @T.prim_func
    def main(
        A: T.Buffer((M, N), dtype),
        B: T.Buffer((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            mype = T.alloc_local([1], "int32")

            mype[0] = T.get_pe()
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, B[by * block_M, bx * block_N])

    return main


func = dist_test(128, 128, 128, 128)

kernel = tilelang.compile(func, out_idx=-1)

# Get CUDA Source
print(kernel.get_kernel_source())

profiler = kernel.get_profiler()
out = profiler.run_once()

print(out)
