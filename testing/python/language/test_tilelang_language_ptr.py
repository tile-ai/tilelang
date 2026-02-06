import torch
from tilelang import tvm as tvm
import tilelang.testing
import tilelang as tl
import tilelang.language as T
from tilelang.utils import map_torch_type


def matmul_test(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        a_ptr: T.ptr,
        b_ptr: T.ptr,
        c_ptr: T.ptr,
        m: T.int32,
        n: T.int32,
        k: T.int32,
    ):
        A = T.make_tensor(a_ptr, (m, k), dtype)
        B = T.make_tensor(b_ptr, (k, n), dtype)
        C = T.make_tensor(c_ptr, (m, n), accum_dtype)

        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(k, block_K), num_stages=3):
                # Copy tile of A
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[bx * block_N, ko * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    program = matmul_test(M, N, K, block_M, block_N, block_K, dtype, accum_dtype)
    jit_kernel = tl.compile(program, target="cuda", execution_backend="cython")

    def ref_program(a, b):
        return (a @ b.T).to(torch.float32)

    a = torch.randn(M, K, device="cuda", dtype=map_torch_type(dtype))
    b = torch.randn(N, K, device="cuda", dtype=map_torch_type(dtype))

    c = torch.zeros(M, N, device="cuda", dtype=map_torch_type(accum_dtype))

    jit_kernel(a.data_ptr(), b.data_ptr(), c.data_ptr(), M, N, K)

    ref_c = (a @ b.T).to(map_torch_type(accum_dtype))

    torch.testing.assert_close(c, ref_c, atol=1e-2, rtol=1e-2)


def test_matmul():
    run_matmul(1024, 1024, 1024, 128, 128, 32)


if __name__ == "__main__":
    tilelang.testing.main()
