# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.

import torch
import tilelang
import tilelang.language as T

def prefix_sum(M, N, blk_m):
    dtype = "float"

    @T.prim_func
    def main(A: T.Buffer((M, N), dtype), B: T.Buffer((M, N), dtype)):
        with T.Kernel(M, threads=128) as bx:
            A_shared = T.alloc_shared((N), dtype)
            tid = T.get_thread_binding()

            T.copy(A[bx * blk_m:(bx + 1) * blk_m, :], A_shared)
            
            steps = T.alloc_var("int32")
            steps = T.log2(T.Cast("float32", N)).astype("int32")
            
            # Up-sweep phase
            for i in T.serial(steps):
                offset = 1 << i
                idx = tid * offset * 2 + offset - 1
                if tid < N // (2 * offset) and idx < N:
                    A_shared[idx + offset] += A_shared[idx]

            if tid == 0:
                A_shared[N - 1] = 0

            # Down-sweep phase
            for i in T.serial(steps):
                offset = N // (1 << (i + 1))
                idx = tid * offset * 2 + offset - 1
                if idx + offset < N:
                    tmp = T.alloc_local([1], dtype)
                    tmp[0] = A_shared[idx + offset]
                    A_shared[idx + offset] += A_shared[idx]
                    A_shared[idx] = tmp[0]

            T.copy(A_shared, B[bx * blk_m:(bx + 1) * blk_m, :])

    return main


def reference_program(x):
    return torch.cumsum(x, dim=1)


def print_sync_barrier_locations(kernel_source):
    """打印CUDA代码中同步障碍的位置"""
    lines = kernel_source.split('\n')
    for i, line in enumerate(lines):
        if "__syncthreads()" in line:
            context_start = max(0, i - 5)
            context_end = min(len(lines), i + 6)
            print(f"发现同步障碍在第 {i+1} 行:")
            for j in range(context_start, context_end):
                if j == i:
                    print(f">>> {lines[j]}")
                else:
                    print(f"    {lines[j]}")
            print()

if __name__ == "__main__":
    M, N, blk_m = 2, 8, 1
    
    print("测试前缀和内核中的线程同步...")
    program = prefix_sum(M, N, blk_m)
    kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
    
    # 检查生成的CUDA代码中的同步屏障
    cuda_source = kernel.get_kernel_source()
    print_sync_barrier_locations(cuda_source)
    
    # 运行内核并验证结果
    a = torch.rand((M, N), device="cuda").float()
    b = torch.zeros_like(a)
    kernel(a, b)
    
    # 与参考实现比较
    ref = reference_program(a)
    torch.testing.assert_close(b, ref, rtol=1e-4, atol=1e-4)
    print("内核输出与PyTorch参考结果匹配。") 