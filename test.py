import tilelang
from tilelang import tvm as tvm
import tilelang.language as T
import torch


from tilelang.engine.callback import register_metal_postproc_callback, register_c_postproc


# @register_metal_postproc_callback
# @register_c_postproc
def print_c_mod(code: str, t) -> str:
    print(code)
    print(t)
    import ipdb

    ipdb.set_trace()
    return code


_cc = tilelang.tvm.contrib.cc._linux_compile

from functools import wraps


@wraps(_cc)
def _patched_cc(output, objects, options, compile_cmd, *args, **kwargs):
    """
    monkey patch to tvm before finalized
    """
    if objects:
        objects = ["-x", "objective-c++"] + objects
    from torch.utils import cpp_extension

    torch_opts = ["-I" + i for i in cpp_extension.include_paths()]
    options += torch_opts + ["-std=gnu++17"]
    return _cc(output, objects, options, compile_cmd, *args, **kwargs)


tilelang.tvm.contrib.cc._linux_compile = _patched_cc


@tilelang.jit(execution_backend="tvm_ffi")
def matmul(M, N, K, block_M, block_N, block_K, dtype=T.float32, accum_dtype=T.float32):
    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype, scope="shared")
            B_shared = T.alloc_shared((block_K, block_N), dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, ko * block_K], A_shared, coalesced_width=2)
                T.copy(B[ko * block_K, bx * block_N], B_shared, coalesced_width=2)

                for i, j, k in T.Parallel(block_M, block_N, block_K):
                    C_local[i, j] += A_shared[i, k] * B_shared[k, j]

            T.copy(C_local, C[by * block_M, bx * block_N], coalesced_width=2)

    return gemm


def assert_gemm(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype=T.float32,
    accum_dtype=T.float32,
    atol=1e-8,
):
    jit_kernel = matmul(M, N, K, block_M, block_N, block_K, dtype=dtype, accum_dtype=accum_dtype)

    torch_dtype = dtype.as_torch()
    a, b = None, None
    if "int" in dtype:
        a = torch.randint(100, (M, K), dtype=torch_dtype, device="mps")
        b = torch.randint(100, (K, N), dtype=torch_dtype, device="mps")
    else:
        a = torch.randn(M, K, dtype=torch_dtype, device="mps")
        b = torch.randn(K, N, dtype=torch_dtype, device="mps")
    c = torch.zeros(M, N, dtype=torch_dtype, device="mps")

    jit_kernel(a, b, c)

    assert torch.allclose(a @ b, c, atol=atol), f"a @ b: {a @ b}, c: {c}"

    assert jit_kernel.kernel_source is not None


if __name__ == "__main__":
    assert_gemm(1024, 1024, 1024, 16, 16, 16, dtype=T.float16, atol=1)
