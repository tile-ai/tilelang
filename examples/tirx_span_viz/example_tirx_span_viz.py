import tilelang
import tilelang.language as T
from tilelang.tools.tirx_span_viz import render_tirx_with_span


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    """Tiled GEMM kernel used to demonstrate span-annotated TIR printing."""
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

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

    return C


def main() -> int:
    """Build the GEMM TIR via ``get_tir`` and print it with source spans."""
    func = matmul.get_tir(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32)
    text = render_tirx_with_span(func)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
