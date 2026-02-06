# Reference: fla/ops/common/chunk_scaled_dot_kkt.py

import tilelang
import tilelang.language as T
import sys  # noqa: F401

# Add your fla repository path to sys.path
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae
# sys.path.insert(0, "/home/tzj/flash-linear-attention")
try:
    import fla

    print(fla.__file__)
    from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
except ImportError:
    print("fla not found, using tilelang implementation")
    fla = None

import torch

torch.set_printoptions(profile="full")
torch.random.manual_seed(0)


def prepare_input(
    B,
    S,
    H,
    DK,
    input_dtype,
    output_dtype,
    accum_dtype,
):
    K = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    Beta = torch.randn(B, S, H, dtype=input_dtype).cuda()
    G = torch.randn(B, S, H, dtype=accum_dtype).cuda()
    return K, Beta, G


def prepare_output(
    B,
    S,
    H,
    chunk_size,
    dtype,
):
    BS = chunk_size
    A = torch.empty(B, S, H, BS, dtype=dtype).cuda()
    return A


@tilelang.jit(out_idx=[-1])
def tilelang_chunk_scaled_dot_kkt_fwd(
    # task config
    B,
    S,
    H,
    DK,
    chunk_size=64,
    input_dtype=T.bfloat16,
    output_dtype=T.bfloat16,
    accum_dtype=T.float32,
    use_g=True,
    # kernel config
    block_S=64,
    block_DK=64,
    threads=256,
    num_stages=0,
):
    K_shape = (B, S, H, DK)
    Beta_shape = (B, S, H)
    G_shape = (B, S, H)
    assert chunk_size == block_S, "chunk_size must be equal to block_S"
    BS = chunk_size
    output_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        K: T.Tensor(K_shape, dtype=input_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=accum_dtype),
        A: T.Tensor(output_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            # !! Pay attention to the scope of the shared memory: may cause misaligned address when shape is one dimension or the buffer is too small
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype, scope="shared")
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            A_shared = T.alloc_shared((block_S, block_S), dtype=output_dtype)
            Beta_K_fragment = T.alloc_fragment((block_S, block_DK), dtype=input_dtype)
            A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            # Tensor used for gated:
            G_shared = T.alloc_shared((block_S,), dtype=accum_dtype, scope="shared")
            G_diff_local = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)

            T.fill(A_fragment, 0)
            T.disable_warp_group_reg_alloc()
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], K_shared)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    Beta_K_fragment[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s]
                T.gemm(Beta_K_fragment, K_shared, A_fragment, transpose_B=True)

            if use_g:
                for i_s in T.Parallel(block_S):
                    G_shared[i_s] = G[bb, bs * block_S + i_s, bh]
                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    G_diff_local[i_s1, i_s2] = G_shared[i_s1] - G_shared[i_s2]
                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    with T.If(G_diff_local[i_s1, i_s2] <= 0 and i_s1 > i_s2):
                        with T.Then():
                            A_fragment[i_s1, i_s2] = A_fragment[i_s1, i_s2] * T.exp(G_diff_local[i_s1, i_s2])
                        with T.Else():
                            A_fragment[i_s1, i_s2] = 0
            else:
                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    with T.If(i_s1 <= i_s2):  # noqa: SIM117
                        with T.Then():
                            A_fragment[i_s1, i_s2] = 0

            T.copy(A_fragment, A_shared)
            T.copy(A_shared, A[bb, bs * block_S : (bs + 1) * block_S, bh, :])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    use_g,
    block_DK,
    threads,
    num_stages,
):
    K, Beta, G = prepare_input(B, S, H, DK, getattr(torch, input_dtype), getattr(torch, output_dtype), getattr(torch, accum_dtype))
    A_ref = prepare_output(B, S, H, chunk_size, getattr(torch, output_dtype))
    A_tilelang = prepare_output(B, S, H, chunk_size, getattr(torch, output_dtype))

    # reference
    if use_g:
        A_ref = chunk_scaled_dot_kkt_fwd(K, Beta, G, chunk_size=chunk_size, output_dtype=getattr(torch, output_dtype))
    else:
        A_ref = chunk_scaled_dot_kkt_fwd(K, Beta, None, chunk_size=chunk_size, output_dtype=getattr(torch, output_dtype))

    # tilelang
    block_S = chunk_size
    kernel = tilelang_chunk_scaled_dot_kkt_fwd(
        B, S, H, DK, chunk_size, input_dtype, output_dtype, accum_dtype, use_g, block_S, block_DK, threads, num_stages
    )
    A_tilelang = kernel(K, Beta, G)

    try:
        torch.testing.assert_close(A_tilelang, A_ref, rtol=1e-2, atol=1e-2)
        print("tilelang chunk scaled dot kkt fwd passed √")
    except Exception as e:
        print("tilelang chunk scaled dot kkt fwd failed ✗")
        print(e)
        print("reference cuda kernel:")
        print(kernel.get_kernel_source())


def main():
    run_test(
        B=1,
        S=32768,
        H=32,
        DK=128,
        chunk_size=64,
        input_dtype=T.bfloat16,
        output_dtype=T.bfloat16,
        accum_dtype=T.float32,
        use_g=True,
        block_DK=64,
        threads=128,
        num_stages=2,
    )


if __name__ == "__main__":
    main()
