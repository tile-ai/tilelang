# Reference: fla/ops/gated_delta_rule/wy_fast.py

import tilelang
import tilelang.language as T
import sys  # noqa: F401

# Add your fla repository path to sys.path
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae
# sys.path.insert(0, "/home/tzj/flash-linear-attention")
try:
    import fla

    print(fla.__file__)
    from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd
except ImportError:
    print("fla not found, using tilelang implementation")
    fla = None

import torch

torch.random.manual_seed(1)


def prepare_input(B, S, H, DK, DV, chunk_size, input_dtype, output_dtype, gate_dtype=torch.float32):
    BS = chunk_size
    K = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    V = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    Beta = torch.randn(B, S, H, dtype=input_dtype).cuda()
    G = torch.randn(B, S, H, dtype=gate_dtype).cuda()
    A = torch.randn(B, S, H, BS, dtype=output_dtype).cuda()
    return K, V, Beta, G, A


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    output_dtype,
):
    W = torch.empty(B, S, H, DK, dtype=output_dtype).cuda()
    U = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return W, U


@tilelang.jit(out_idx=[-2, -1])
def tilelang_recompute_w_u_fwd(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    gate_dtype,
    accum_dtype,
    chunk_size,
    # kernel config
    block_S=64,
    block_DK=64,
    block_DV=64,
    threads=256,
    num_stages=0,
):
    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    Beta_shape = (B, S, H)
    assert chunk_size == block_S, "chunk_size must be equal to block_S"
    BS = chunk_size
    G_shape = (B, S, H)
    A_shape = (B, S, H, BS)

    @T.prim_func
    def kernel(
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        Beta: T.Tensor(Beta_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        A: T.Tensor(A_shape, dtype=output_dtype),
        W: T.Tensor(K_shape, dtype=output_dtype),
        U: T.Tensor(V_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(S, block_S), B * H, threads=threads) as (bs, bbh):
            bb, bh = bbh // H, bbh % H
            Beta_shared = T.alloc_shared((block_S,), dtype=input_dtype, scope="shared")
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            G_shared = T.alloc_shared((block_S,), dtype=gate_dtype, scope="shared")
            A_shared = T.alloc_shared((block_S, block_S), dtype=output_dtype)
            W_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            W_shared = T.alloc_shared((block_S, block_DK), dtype=output_dtype)
            U_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            W_Beta_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            U_Beta_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)

            T.annotate_layout(
                {
                    K_shared: tilelang.layout.make_swizzled_layout(K_shared),
                    V_shared: tilelang.layout.make_swizzled_layout(V_shared),
                }
            )

            T.disable_warp_group_reg_alloc()
            for i_s in T.Parallel(block_S):
                Beta_shared[i_s] = Beta[bb, bs * block_S + i_s, bh]
                G_shared[i_s] = T.exp(G[bb, bs * block_S + i_s, bh])

            T.copy(A[bb, bs * block_S : (bs + 1) * block_S, bh, :], A_shared)

            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(V[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], V_shared)
                for i_s, i_v2 in T.Parallel(block_S, block_DV):
                    U_Beta_shared[i_s, i_v2] = V_shared[i_s, i_v2] * Beta_shared[i_s]
                T.gemm(A_shared, U_Beta_shared, U_fragment, clear_accum=True)
                # First copy to smem, then copy to gmem to reduce U2RU instructions
                T.copy(U_fragment, U_shared)
                T.copy(U_shared, U[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV])

            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], K_shared)
                for i_s, i_k2 in T.Parallel(block_S, block_DK):
                    W_Beta_shared[i_s, i_k2] = K_shared[i_s, i_k2] * Beta_shared[i_s] * G_shared[i_s]
                T.gemm(A_shared, W_Beta_shared, W_fragment, clear_accum=True)
                # First copy to smem, then copy to gmem to reduce U2RU instructions
                T.copy(W_fragment, W_shared)
                T.copy(W_shared, W[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK])

    return kernel


def run_test(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    output_dtype,
    gate_dtype,
    accum_dtype,
    block_DK,
    block_DV,
    threads,
    num_stages,
):
    K, V, Beta, G, A = prepare_input(
        B, S, H, DK, DV, chunk_size, getattr(torch, input_dtype), getattr(torch, output_dtype), gate_dtype=getattr(torch, gate_dtype)
    )
    W_ref, U_ref = prepare_output(B, S, H, DK, DV, getattr(torch, output_dtype))
    W_tilelang, U_tilelang = prepare_output(B, S, H, DK, DV, getattr(torch, output_dtype))

    # reference
    W_ref, U_ref = recompute_w_u_fwd(K, V, Beta, G, A, None)

    # tilelang
    block_S = chunk_size
    kernel = tilelang_recompute_w_u_fwd(
        B,
        S,
        H,
        DK,
        DV,
        input_dtype,
        output_dtype,
        gate_dtype,
        accum_dtype,
        chunk_size,
        block_S=block_S,
        block_DK=block_DK,
        block_DV=block_DV,
        threads=threads,
        num_stages=num_stages,
    )
    print(kernel.get_kernel_source())
    W_tilelang, U_tilelang = kernel(K, V, Beta, G, A)

    try:
        torch.testing.assert_close(W_tilelang, W_ref, rtol=1e-2, atol=1e-2)
        print("tilelang recompute w passed √")
    except Exception as e:
        print("tilelang recompute w failed ✗")
        print(e)
    try:
        torch.testing.assert_close(U_tilelang, U_ref, rtol=1e-2, atol=1e-2)
        print("tilelang recompute u passed √")
    except Exception as e:
        print("tilelang recompute u failed ✗")
        print(e)


def main():
    run_test(
        B=1,
        S=32768,
        H=32,
        DK=128,
        DV=128,
        chunk_size=64,
        input_dtype=T.bfloat16,
        output_dtype=T.bfloat16,
        gate_dtype=T.float32,
        accum_dtype=T.float32,
        block_DK=64,
        block_DV=32,
        threads=128,
        num_stages=3,
    )


if __name__ == "__main__":
    main()
