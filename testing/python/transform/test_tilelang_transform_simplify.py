from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def modify(
    with_B: bool = False,
    with_bias: bool = False,
):
    @T.prim_func
    def main(
        A: T.Tensor((64, 64)),
        B: T.Tensor((64, 64)),
        C: T.Tensor((64, 64)),
        D: T.Tensor((64, 64)),
        bias: T.Tensor((64, 64)),
    ):
        if with_B:
            if with_bias:
                T.gemm(A, bias, D)
            T.gemm(A, B, D)
        else:
            with T.block():
                A_shared = T.alloc_shared((64, 64), dtype=T.float32)
                C_shared = T.alloc_shared((64, 64), dtype=T.float32)
                D_shared = T.alloc_shared((64, 64), dtype=T.float32)
                T.copy(A, A_shared)
                T.copy(C, C_shared)
                T.gemm(A_shared, C_shared, D_shared)
                T.copy(D_shared, D)

    return main


def test_modify(with_B=False, with_bias=False):
    tester = modify(with_B=with_B, with_bias=with_bias)
    mod = tvm.IRModule({tester.attrs["global_symbol"]: tester})
    mod2 = tl.transform.Simplify()(mod)
    assert mod != mod2


def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    @T.prim_func
    def main(
        a: T.handle,
        b: T.handle,
        c: T.handle,
    ):
        A = T.match_buffer(a, (M, K), dtype=dtype)
        B = T.match_buffer(b, (K, N), dtype=dtype)
        C = T.match_buffer(c, (M, N), dtype=accum_dtype)

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def test_matmul():
    func = matmul(1024, 1024, 1024, 128, 128, 32)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    mod = tl.transform.Simplify()(mod)
    kernel = tl.compile(mod["main"], out_idx=[2])

    import torch

    a = torch.randn(1024, 1024, dtype=torch.float16).cuda().half()
    b = torch.randn(1024, 1024, dtype=torch.float16).cuda().half()
    c = kernel(a, b)

    ref_c = a @ b
    ref_c = ref_c.float()
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)

    # Get CUDA Source
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    tilelang.testing.main()
