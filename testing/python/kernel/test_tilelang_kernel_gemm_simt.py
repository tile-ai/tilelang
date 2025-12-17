import torch
import torch.backends
import tilelang.testing
from tilelang import tvm as tvm
from tvm import DataType
import tilelang.language as T
from tilelang.intrinsics import get_swizzle_layout
from tilelang.transform import simplify_prim_func

tilelang.testing.set_random_seed(0)


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        return T.Layout(shape, lambda *args: args)

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


@simplify_prim_func
def tl_matmul_simt(
    M,
    N,
    K,
    in_dtype,
    out_dtype,
    accum_dtype,
):
    assert in_dtype in [
        T.float16,
        T.int8,
    ], "Currently only float16 and int8 are supported"
    assert out_dtype in [
        T.float16,
        T.float32,
        T.int32,
    ], "Currently only float16, float32 and int32 are supported"

    # This is a debug config
    block_size_x = 8
    block_size_y = 8
    thread_row_tiles = 16
    thread_col_tiles = 16
    chunk = 16

    shared_scope = "shared"

    block_M = block_size_x * thread_row_tiles
    block_N = block_size_y * thread_col_tiles
    block_K = chunk

    # Pipeline Stage

    A_shape = (M, K)
    B_shape = (N, K)
    C_shape = (M, N)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)

    threads = thread_row_tiles * thread_col_tiles
    local_size_a = block_M // thread_row_tiles
    local_size_b = block_N // thread_col_tiles
    local_size_c = (block_M // thread_row_tiles) * (block_N // thread_col_tiles)

    micro_size_k = 128 // DataType(in_dtype).bits
    dp4a_size = 4
    use_dp4a = in_dtype == T.int8 and accum_dtype == T.int32

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor(C_shape, out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope=shared_scope)

            A_local = T.alloc_local((local_size_a, micro_size_k), in_dtype)
            B_local = T.alloc_local((local_size_b, micro_size_k), in_dtype)
            C_local = T.alloc_local((local_size_c,), accum_dtype)

            tid = T.get_thread_binding()

            warp_m = tid % thread_row_tiles
            warp_n = tid // thread_row_tiles

            T.clear(C_local)

            for ko in T.serial(K // block_K):
                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for ki in T.serial((block_K // micro_size_k)):
                    for i in T.serial(local_size_a):
                        for mk in T.vectorized(micro_size_k):
                            A_local[i, mk] = A_shared[warp_m * local_size_a + i, ki * micro_size_k + mk]

                    for i in T.serial(local_size_b):
                        for mk in T.vectorized(micro_size_k):
                            B_local[i, mk] = B_shared[warp_n * local_size_b + i, ki * micro_size_k + mk]

                    for i, j in T.grid(local_size_a, local_size_b):
                        for mk in T.serial(micro_size_k // dp4a_size):
                            if use_dp4a:
                                T.dp4a(A_local[i, mk * dp4a_size], B_local[j, mk * dp4a_size], C_local[i * local_size_b + j])
                            else:
                                for dp4a_idx in T.serial(dp4a_size):
                                    C_local[i * local_size_b + j] += (
                                        A_local[i, mk * dp4a_size + dp4a_idx] * B_local[j, mk * dp4a_size + dp4a_idx]
                                    )

            for i, j in T.grid(local_size_a, local_size_b):
                C[by * block_M + warp_m * local_size_a + i, bx * block_N + warp_n * local_size_b + j] = C_local[i * local_size_b + j]

    return main


def assert_tl_matmul_correctness(M, N, K, in_dtype, out_dtype, accum_dtype):
    matmul = tl_matmul_simt(M, N, K, in_dtype, out_dtype, accum_dtype)
    kernel = tilelang.compile(matmul, out_idx=[2])
    profiler = kernel.get_profiler()

    src_code = kernel.get_kernel_source()
    print(src_code)
    # src_code is the generated cuda source
    assert src_code is not None

    if in_dtype == T.int8:
        A = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
        B = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8)
    else:
        A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
        B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))

    C = kernel(A, B)

    latency = profiler.do_bench()

    # Ensure that the latency is not None
    assert latency is not None

    # Get Reference Result
    ref_c = torch.matmul(A.to(torch.float32), B.T.to(torch.float32)).to(getattr(torch, accum_dtype))
    print(C)
    print(ref_c)
    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def test_assert_tl_matmul():
    assert_tl_matmul_correctness(128, 128, 128, T.float16, T.float16, T.float16)
    assert_tl_matmul_correctness(128, 256, 256, T.float16, T.float32, T.float32)
    assert_tl_matmul_correctness(128, 256, 256, T.int8, T.int32, T.int32)


if __name__ == "__main__":
    tilelang.testing.main()
