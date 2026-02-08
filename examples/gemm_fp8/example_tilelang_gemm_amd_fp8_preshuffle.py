import torch
import itertools
import tilelang
import tilelang.testing
from tilelang import tvm as tvm
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.layout import make_swizzled_layout
from tilelang.intrinsics.mfma_macro_generator import MatrixCorePreshuffleIntrinEmitter
from tilelang.utils import determine_fp8_type

tilelang.testing.set_random_seed(0)


def get_configs():
    block_Ms = [32, 64, 128]
    block_Ns = [32, 64, 128]
    block_Ks = [64, 128]
    num_stages = [0, 1, 2]

    valid_configs = []

    for m, n, k, stages in itertools.product(block_Ms, block_Ns, block_Ks, num_stages):
        valid_configs.append(
            {
                "block_M": m,
                "block_N": n,
                "block_K": k,
                "num_stages": stages,
            }
        )
    return valid_configs


@tilelang.autotune(
    configs=get_configs(),
)
@tilelang.jit(out_idx=[-1])
def tl_matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    num_stages,
    k_pack=2,
    num_threads=256,
    in_dtype=None,
    out_dtype=T.float32,
    accum_dtype=T.float32,
    a_transposed=False,
    b_transposed=True,
):
    if in_dtype is None:
        in_dtype = determine_fp8_type()
    b_preshuffle = True
    warp_size = 64
    num_warps = num_threads // warp_size

    policy = GemmWarpPolicy.Square
    m_warp, n_warp = policy.compute_warp_partition(block_M, block_N, num_warps)

    shared_scope = "shared"
    warp_row_tiles = block_M // m_warp
    warp_col_tiles = block_N // n_warp

    # MMA Wrapper to Auto Generate Code for MMA
    mfma_emitter = MatrixCorePreshuffleIntrinEmitter(
        a_dtype=in_dtype,
        b_dtype=in_dtype,
        accum_dtype=accum_dtype,
        a_transposed=a_transposed,
        b_transposed=b_transposed,
        block_row_warps=m_warp,
        block_col_warps=n_warp,
        warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles,
        chunk=block_K,
        k_pack=k_pack,
        b_preshuffle=b_preshuffle,
    )
    local_size_a = mfma_emitter.local_size_a
    local_size_b = mfma_emitter.local_size_b

    warp_rows = mfma_emitter.warp_rows
    warp_cols = mfma_emitter.warp_cols

    micro_size_y = mfma_emitter.micro_size_y
    micro_size_k = mfma_emitter.micro_size_k
    pack_size_k = micro_size_k * k_pack

    A_shape = (K, M) if a_transposed else (M, K)
    A_shared_shape = (block_K, block_M) if a_transposed else (block_M, block_K)

    B_shape = (
        (N // micro_size_y, K // pack_size_k, micro_size_y, pack_size_k)
        if b_transposed
        else (K // pack_size_k, N // micro_size_y, pack_size_k, micro_size_y)
    )

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=num_threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope=shared_scope)
            A_local = T.alloc_local((warp_rows * local_size_a * k_pack), in_dtype)
            B_local = T.alloc_local((warp_cols * local_size_b * k_pack), in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.annotate_layout(
                {
                    A_shared: make_swizzled_layout(A_shared),
                    C_local: mfma_emitter.make_mfma_store_layout(C_local),
                }
            )

            num_ko = K // block_K
            num_ki = block_K // (k_pack * micro_size_k)

            # Improve L2 Cache
            # T.use_swizzle(panel_size=10)
            T.clear(C_local)
            for ko in T.Pipelined(num_ko, num_stages=num_stages):
                # Load A into shared memory
                if a_transposed:
                    T.copy(A[ko * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, ko * block_K], A_shared)

                for ki in T.serial(0, num_ki):
                    mfma_emitter.ldmatrix_a(
                        A_local,
                        A_shared,
                        ki,
                    )
                    mfma_emitter.ldmatrix_b(B_local, B, ki + ko * num_ki, pid_m=by, pid_n=bx)

                    # Perform Matrix Multiplication
                    mfma_emitter.mfma(A_local, B_local, C_local, ki)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def shuffle_weight(
    x: torch.Tensor,
    layout=(16, 32),
    k_pack=1,
    is_transpose=False,
) -> torch.Tensor:
    IN, IK = layout
    BK = IK * k_pack
    BN = IN

    N, K = (x.shape[-2], x.shape[-1]) if is_transpose else (x.shape[-1], x.shape[-2])
    assert N % BN == 0
    assert K % BK == 0

    x = x.view(N // BN, BN, K // BK, BK) if is_transpose else x.view(K // BK, BK, N // BN, BN)
    x = x.permute(0, 2, 1, 3)
    return x.contiguous()


def assert_tl_matmul_correctness(M, N, K, k_pack=1, a_transposed=False, b_transposed=True):
    in_dtype = determine_fp8_type()
    out_dtype = T.float32
    accum_dtype = T.float32
    kernel = tl_matmul(
        M,
        N,
        K,
        k_pack=k_pack,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        accum_dtype=accum_dtype,
        a_transposed=a_transposed,
        b_transposed=b_transposed,
    )

    src_code = kernel.get_kernel_source()
    # src_code is the generated cuda source
    assert src_code is not None
    A_shape = (K, M) if a_transposed else (M, K)
    B_shape = (N, K) if b_transposed else (K, N)

    A = (torch.rand(A_shape, device="cuda", dtype=torch.float16) / 10).to(getattr(torch, in_dtype))
    B = (torch.rand(B_shape, device="cuda", dtype=torch.float16) / 10).to(getattr(torch, in_dtype))

    B_preshuffle = shuffle_weight(B, k_pack=k_pack, is_transpose=b_transposed)
    C = kernel(A, B_preshuffle)

    profiler = kernel.get_profiler()
    latency = profiler.do_bench()

    # Ensure that the latency is not None
    assert latency is not None
    print("time: ", latency)

    if a_transposed and b_transposed:
        # Get Reference Result
        ref_c = torch.matmul(A.T.half(), B.T.half()).to(getattr(torch, out_dtype))
    elif a_transposed and not b_transposed:
        # Get Reference Result
        ref_c = torch.matmul(A.T.half(), B.half()).to(getattr(torch, out_dtype))
    elif not a_transposed and b_transposed:
        # Get Reference Result
        ref_c = torch.matmul(A.half(), B.T.half()).to(getattr(torch, out_dtype))
    else:
        # Get Reference Result
        ref_c = torch.matmul(A.half(), B.half()).to(getattr(torch, out_dtype))

    torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)


def test_assert_tl_matmul():
    assert_tl_matmul_correctness(512, 512, 512, k_pack=2)


if __name__ == "__main__":
    test_assert_tl_matmul()
