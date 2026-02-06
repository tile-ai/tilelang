# Reference: fla/ops/common/chunk_o.py

import tilelang
import tilelang.language as T
import sys  # noqa: F401

# Add your fla repository path to sys.path
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae
# sys.path.insert(0, "/home/tzj/flash-linear-attention")
try:
    import fla

    print(fla.__file__)
    from fla.ops.common.chunk_o import chunk_fwd_o
except ImportError:
    print("fla not found, using tilelang implementation")
    fla = None

import torch

torch.random.manual_seed(1)


def prepare_input(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
):
    BS = chunk_size
    Q = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    K = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    V = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    HIDDEN = torch.randn(B, S // BS, H, DK, DV, dtype=input_dtype).cuda()
    G = torch.randn(B, S, H, dtype=gate_dtype).cuda()
    return Q, K, V, HIDDEN, G


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    output_dtype,
):
    O = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return O


@tilelang.jit(out_idx=[-1])
def tilelang_chunk_fwd_o(
    # task config
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    chunk_size,
    scale,
    use_g,
    # kernel config
    block_S=64,
    block_DK=64,
    block_DV=64,
    threads=256,
    num_stages=0,
):
    assert chunk_size == block_S, "chunk_size must be equal to block_S"
    BS = chunk_size
    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    H_shape = (B, S // BS, H, DK, DV)
    G_shape = (B, S, H)
    O_shape = (B, S, H, DV)

    @T.prim_func
    def kernel(
        Q: T.Tensor(Q_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        HIDDEN: T.Tensor(H_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        O: T.Tensor(O_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), T.ceildiv(S, block_S), B * H, threads=threads) as (bv, bs, bbh):
            bb, bh = bbh // H, bbh % H
            Q_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            K_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            H_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            A_shared = T.alloc_shared((block_S, block_S), dtype=input_dtype)
            O_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            A_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            O_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            G_shared = T.alloc_shared((block_S,), dtype=gate_dtype, scope="shared")
            G_diff_local = T.alloc_fragment((block_S, block_S), dtype=gate_dtype)

            T.clear(A_fragment)
            T.clear(O_fragment)
            T.disable_warp_group_reg_alloc()
            for i_k in T.Pipelined(T.ceildiv(DK, block_DK), num_stages=num_stages):
                T.copy(Q[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], Q_shared)
                T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, i_k * block_DK : (i_k + 1) * block_DK], K_shared)
                T.copy(HIDDEN[bb, bs, bh, i_k * block_DK : (i_k + 1) * block_DK, bv * block_DV : (bv + 1) * block_DV], H_shared)
                T.gemm(Q_shared, H_shared, O_fragment)
                T.gemm(Q_shared, K_shared, A_fragment, transpose_B=True)

            if use_g:
                for i_s in T.Parallel(block_S):
                    G_shared[i_s] = G[bb, bs * block_S + i_s, bh]
                # T.copy(G[bb, bs * block_S:(bs + 1) * block_S, bh], G_shared)
                for i_s, i_v in T.Parallel(block_S, block_DV):
                    O_fragment[i_s, i_v] = O_fragment[i_s, i_v] * T.exp(G_shared[i_s])
                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    G_diff_local[i_s1, i_s2] = G_shared[i_s1] - G_shared[i_s2]
                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    with T.If(G_diff_local[i_s1, i_s2] <= 0):
                        with T.Then():
                            A_fragment[i_s1, i_s2] = A_fragment[i_s1, i_s2] * T.exp(G_diff_local[i_s1, i_s2])
                        with T.Else():
                            A_fragment[i_s1, i_s2] = 0

            for i_s1, i_s2 in T.Parallel(block_S, block_S):
                with T.If(i_s1 < i_s2):  # noqa: SIM117
                    with T.Then():
                        A_fragment[i_s1, i_s2] = 0

            T.copy(V[bb, bs * block_S : (bs + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV], V_shared)
            T.copy(A_fragment, A_shared)
            T.gemm(A_shared, V_shared, O_fragment)

            for i_s, i_v in T.Parallel(block_S, block_DV):
                O_fragment[i_s, i_v] = O_fragment[i_s, i_v] * scale

            T.copy(O_fragment, O_shared)
            T.copy(O_shared, O[bb, bs * block_S : (bs + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV])

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
    accum_dtype,
    gate_dtype,
    use_g,
    block_DK,
    block_DV,
    threads,
    num_stages,
):
    input_dtype_torch = getattr(torch, input_dtype)
    output_dtype_torch = getattr(torch, output_dtype)
    accum_dtype_torch = getattr(torch, accum_dtype)
    gate_dtype_torch = getattr(torch, gate_dtype)
    Q, K, V, HIDDEN, G = prepare_input(
        B, S, H, DK, DV, chunk_size, input_dtype_torch, output_dtype_torch, accum_dtype_torch, gate_dtype_torch
    )
    scale = 1.0 / DK**0.5

    O_ref = prepare_output(B, S, H, DK, DV, chunk_size, output_dtype_torch)
    O_ref = chunk_fwd_o(Q, K, V, HIDDEN, G, scale, chunk_size=chunk_size)

    block_S = chunk_size
    O_tilelang = prepare_output(B, S, H, DK, DV, chunk_size, output_dtype_torch)
    kernel = tilelang_chunk_fwd_o(
        B,
        S,
        H,
        DK,
        DV,
        input_dtype,
        output_dtype,
        accum_dtype,
        gate_dtype,
        chunk_size,
        scale,
        use_g,
        block_S,
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    O_tilelang = kernel(Q, K, V, HIDDEN, G)

    try:
        torch.testing.assert_close(O_tilelang, O_ref, rtol=1e-2, atol=1e-2)
        print("tilelang chunk fwd o passed √")
    except Exception as e:
        print("tilelang chunk fwd o failed ✗")
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
        accum_dtype=T.float32,
        gate_dtype=T.float32,
        use_g=True,
        block_DK=128,
        block_DV=128,
        threads=128,
        num_stages=1,
    )


if __name__ == "__main__":
    main()
