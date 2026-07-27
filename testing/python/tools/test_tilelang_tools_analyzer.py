import tilelang
import tilelang.language as T
from tilelang.tools import Analyzer


class _Device:
    compute_capability = "80"
    bandwidth = [750, 12080]


@tilelang.jit
def _gemm(A, B):
    M = N = K = 64
    block_M = block_N = block_K = 32

    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((N, K), T.float16)
    C = T.empty((M, N), T.float16)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float16)
        B_shared = T.alloc_shared((block_N, block_K), T.float16)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)

        T.clear(C_local)
        for k in T.serial(T.ceildiv(K, block_K)):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


@tilelang.jit
def _single_block_gemm(A, B):
    M = N = K = 32

    A: T.Tensor((M, K), T.float16)
    B: T.Tensor((N, K), T.float16)
    C = T.empty((M, N), T.float16)

    with T.Kernel(1, threads=128):
        A_shared = T.alloc_shared((M, K), T.float16)
        B_shared = T.alloc_shared((N, K), T.float16)
        C_local = T.alloc_fragment((M, N), T.float32)

        T.copy(A, A_shared)
        T.copy(B, B_shared)
        T.clear(C_local)
        T.gemm(A_shared, B_shared, C_local, transpose_B=True)
        T.copy(C_local, C)

    return C


def test_analyzer_recognizes_tileop_names():
    result = Analyzer.analysis(_gemm.get_tir(), _Device())

    assert result.total_flops == 2 * 64 * 64 * 64
    assert result.total_global_bytes == 40_960


def test_analyzer_resets_grid_extents_for_each_prim_func():
    analyzer = Analyzer(_single_block_gemm.get_tir(), _Device())
    analyzer.block_counts["blockIdx.y"] = 7

    result = analyzer.ir_pass().calculate()

    assert result.total_flops == 2 * 32 * 32 * 32
    assert result.total_global_bytes == 6_144
