# Reference: fla/ops/common/chunk_o.py

import math
import sys  # noqa: F401

import tilelang
import tilelang.language as T
from tilelang.engine.callback import register_cuda_postproc_callback  # noqa: F401

# Add your fla repository path to sys.path
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae
# sys.path.insert(0, "/home/tzj/flash-linear-attention")
try:
    import fla

    print(fla.__file__)
    from fla.ops.common.chunk_o import chunk_bwd_dqkwg
except ImportError:
    print("fla not found, using tilelang implementation")
    fla = None

import torch
from test_utils import assert_similar

torch.random.manual_seed(0)
# torch.set_printoptions(profile="full")


def prepare_input_fake(
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
    state_dtype,
):
    BS = S // chunk_size
    Q = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    K = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    V = torch.ones(B, S, H, DV, dtype=input_dtype).cuda()
    h = torch.ones(B, BS, H, DK, DV, dtype=input_dtype).cuda()
    G = torch.ones(B, S, H, dtype=gate_dtype).cuda()
    dO = torch.ones(B, S, H, DV, dtype=input_dtype).cuda()
    dh = torch.ones(B, BS, H, DK, DV, dtype=input_dtype).cuda()
    dv = torch.ones(B, S, H, DV, dtype=output_dtype).cuda()
    W = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    return Q, K, V, h, G, dO, dh, dv, W


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
    state_dtype,
):
    BS = S // chunk_size

    Q = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    K = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    V = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    h = torch.randn(B, BS, H, DK, DV, dtype=input_dtype).cuda()
    G = torch.randn(B, S, H, dtype=gate_dtype).cuda()
    dO = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    dh = torch.randn(B, BS, H, DK, DV, dtype=input_dtype).cuda()
    dv = torch.randn(B, S, H, DV, dtype=output_dtype).cuda()
    W = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    return Q, K, V, h, G, dO, dh, dv, W


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    output_dtype,
    gate_dtype,
    state_dtype,
    block_DK,
):
    assert DK == 32 and block_DK == 32 or DK > 32 and block_DK >= 64, "When DK > 32, block_DK must be >= 64"
    NK = math.ceil(DK / block_DK)
    dq = torch.empty(B, S, H, DK, dtype=output_dtype).cuda()
    dk = torch.empty(B, S, H, DK, dtype=output_dtype).cuda()
    dw = torch.empty(B, S, H, DK, dtype=output_dtype).cuda()
    dg = torch.empty(NK, B, S, H, dtype=gate_dtype).cuda()
    return dq, dk, dw, dg


# @register_cuda_postproc_callback
# def tilelang_callback_cuda_postproc(code, _):
#     cuda_code = open("../debug/chunk_o_bwd3.log", "r").read()
#     code = cuda_code
#     return code


@tilelang.jit(
    out_idx=[-4, -3, -2, -1],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def tilelang_chunk_o_bwd_dqkwg(
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
    state_dtype,
    chunk_size,
    scale,
    use_g=True,
    use_dw=True,
    # kernel config
    block_DK=64,
    block_DV=64,
    threads=256,
    num_stages=0,
):
    block_S = chunk_size
    BS = S // block_S
    NK = math.ceil(DK / block_DK)

    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    h_shape = (B, BS, H, DK, DV)
    G_shape = (B, S, H)
    dO_shape = (B, S, H, DV)
    dh_shape = (B, BS, H, DK, DV)
    dv_shape = (B, S, H, DV)
    W_shape = (B, S, H, DK)

    dq_shape = (B, S, H, DK)
    dk_shape = (B, S, H, DK)
    dw_shape = (B, S, H, DK)
    dg_shape = (NK, B, S, H)

    @T.prim_func
    def kernel(
        # input
        Q: T.Tensor(Q_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        V: T.Tensor(V_shape, dtype=input_dtype),
        h: T.Tensor(h_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        dO: T.Tensor(dO_shape, dtype=input_dtype),
        dh: T.Tensor(dh_shape, dtype=input_dtype),
        dv: T.Tensor(dv_shape, dtype=input_dtype),
        W: T.Tensor(W_shape, dtype=input_dtype),
        # output
        dq: T.Tensor(dq_shape, dtype=output_dtype),
        dk: T.Tensor(dk_shape, dtype=output_dtype),
        dw: T.Tensor(dw_shape, dtype=output_dtype),
        dg: T.Tensor(dg_shape, dtype=gate_dtype),
    ):
        with T.Kernel(T.ceildiv(DK, block_DK), T.ceildiv(S, block_S), B * H, threads=threads) as (bk, bs, bbh):
            bb, bh = bbh // H, bbh % H

            V_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            dO_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            h_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            dh_shared = T.alloc_shared((block_DK, block_DV), dtype=input_dtype)
            dv_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            q_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            k_shared = T.alloc_shared((block_S, block_DK), dtype=input_dtype)
            ds_shared = T.alloc_shared((block_S, block_S), dtype=output_dtype)
            dg_shared_1 = T.alloc_shared((block_S,), dtype=gate_dtype)
            dg_shared_2 = T.alloc_shared((block_S,), dtype=gate_dtype)
            dk_shared = T.alloc_shared((block_S, block_DK), dtype=accum_dtype)

            ds_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            ds_fragment_positive = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            ds_fragment_positive_transpose = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            dq_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dk_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dk_fragment_2 = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            dw_fragment = T.alloc_fragment((block_S, block_DK), dtype=accum_dtype)
            q_fragment = T.alloc_fragment((block_S, block_DK), dtype=input_dtype)
            k_fragment = T.alloc_fragment((block_S, block_DK), dtype=input_dtype)

            dg_fragment_reduce_tmp = T.alloc_fragment((block_S, block_DK), dtype=gate_dtype)
            dg_fragment = T.alloc_fragment((block_S,), dtype=gate_dtype)
            dg_fragment_2 = T.alloc_fragment((block_S,), dtype=gate_dtype)
            dg_fragment_final = T.alloc_fragment((block_S,), dtype=gate_dtype)
            dg_last_local_0 = T.alloc_var(dtype=gate_dtype)
            dg_last_local_1 = T.alloc_var(dtype=gate_dtype)
            G_last_local = T.alloc_var(dtype=gate_dtype)

            dg_last_fragment = T.alloc_fragment((block_DV * block_DK), dtype=gate_dtype)
            dg_last_fragment_scalar = T.alloc_fragment((1,), dtype=gate_dtype)
            dg_last_fragment_2 = T.alloc_fragment((block_S * block_DK), dtype=gate_dtype)
            dg_last_fragment_scalar_2 = T.alloc_fragment((1,), dtype=gate_dtype)
            G_shared = T.alloc_shared((block_S, block_DK), dtype=gate_dtype)

            T.use_swizzle(10)

            T.annotate_layout(
                {
                    q_shared: tilelang.layout.make_swizzled_layout(q_shared),
                    k_shared: tilelang.layout.make_swizzled_layout(k_shared),
                }
            )

            T.clear(dg_last_local_0)
            T.clear(dg_last_local_1)
            T.clear(G_last_local)
            T.clear(G_shared)
            T.clear(q_fragment)
            T.clear(k_fragment)
            T.clear(dg_last_fragment)

            T.clear(ds_fragment)
            T.clear(dq_fragment)
            T.clear(dk_fragment)
            T.clear(dw_fragment)

            for i_v in T.Pipelined(T.ceildiv(DV, block_DV), num_stages=num_stages):
                T.copy(V[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], V_shared)
                T.copy(dO[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], dO_shared)
                T.copy(h[bb, bs, bh, bk * block_DK : (bk + 1) * block_DK, i_v * block_DV : (i_v + 1) * block_DV], h_shared)
                T.copy(dh[bb, bs, bh, bk * block_DK : (bk + 1) * block_DK, i_v * block_DV : (i_v + 1) * block_DV], dh_shared)

                if use_g:
                    T.clear(dg_last_fragment_scalar)
                    # FIXME: The reduce operation of a whole buffer to a scalar is not supported and will cause incorrect result
                    # for i_kv in T.Parallel(block_DK * block_DV):
                    #     dg_last_fragment[i_kv] = h_shared[i_kv // block_DV, i_kv % block_DV] * dh_shared[i_kv // block_DV, i_kv % block_DV]
                    for i_kv in T.Parallel(block_DK * block_DV):
                        dg_last_fragment[i_kv] = h_shared[i_kv // block_DV, i_kv % block_DV] * dh_shared[i_kv // block_DV, i_kv % block_DV]
                    T.reduce_sum(dg_last_fragment, dg_last_fragment_scalar, dim=-1, clear=False)
                    dg_last_local_0 = dg_last_local_0 + dg_last_fragment_scalar[0]

                T.gemm(dO_shared, V_shared, ds_fragment, transpose_B=True)
                T.gemm(dO_shared, h_shared, dq_fragment, transpose_B=True)
                T.gemm(V_shared, dh_shared, dk_fragment, transpose_B=True)

                if use_dw:
                    T.copy(dv[bb, bs * block_S : (bs + 1) * block_S, bh, i_v * block_DV : (i_v + 1) * block_DV], dv_shared)
                    T.gemm(dv_shared, h_shared, dw_fragment, transpose_B=True)

            if use_dw:
                for i_s, i_k in T.Parallel(block_S, block_DK):
                    dw_fragment[i_s, i_k] = -dw_fragment[i_s, i_k]
                T.copy(dw_fragment, dw[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK])

            T.copy(Q[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK], q_shared)
            T.copy(K[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK], k_shared)
            T.copy(q_shared, q_fragment)
            T.copy(k_shared, k_fragment)

            if use_g:
                T.clear(dg_fragment)
                T.clear(dg_fragment_2)
                for i_s, i_k in T.Parallel(block_S, block_DK):
                    G_shared[i_s, i_k] = G[bb, bs * block_S + i_s, bh]
                dg_last_local_0 = G[bb, bs * block_S + block_S - 1, bh]
                # Use gmem directly instead of local register
                dg_last_local_0 = dg_last_local_0 * T.exp(G[bb, bs * block_S + block_S - 1, bh])

                for i_s, i_k in T.Parallel(block_S, block_DK):
                    dq_fragment[i_s, i_k] = dq_fragment[i_s, i_k] * T.exp(G[bb, bs * block_S + i_s, bh]) * scale
                T.clear(dg_fragment_reduce_tmp)
                for i_s, i_k in T.Parallel(block_S, block_DK):
                    dg_fragment_reduce_tmp[i_s, i_k] = dq_fragment[i_s, i_k] * q_shared[i_s, i_k]
                # FIXME: The reduce_sum statement with clear=True will cause an error of warp specialized pass
                T.reduce_sum(dg_fragment_reduce_tmp, dg_fragment, dim=-1, clear=False)

                for i_s, i_k in T.Parallel(block_S, block_DK):
                    dk_fragment[i_s, i_k] = (
                        dk_fragment[i_s, i_k] * T.exp(G_last_local - G[bb, bs * block_S + i_s, bh])
                        if G_last_local - G[bb, bs * block_S + i_s, bh] <= 0
                        else 0
                    )
                T.clear(dg_fragment_reduce_tmp)
                for i_s, i_k in T.Parallel(block_S, block_DK):
                    dg_fragment_reduce_tmp[i_s, i_k] = dk_fragment[i_s, i_k] * (-k_shared[i_s, i_k])
                # FIXME: The reduce_sum statement with clear=True will cause an error of warp specialized pass
                T.reduce_sum(dg_fragment_reduce_tmp, dg_fragment, dim=-1, clear=False)

                # FIXME: The reduce operation of a whole buffer to a scalar is not supported and will cause incorrect result
                T.copy(dk_fragment, dk_shared)
                T.clear(dg_last_fragment_scalar_2)
                for i_sk in T.Parallel(block_S * block_DK):
                    i_s, i_k = i_sk // block_DK, i_sk % block_DK
                    dg_last_fragment_2[i_sk] = dk_shared[i_s, i_k] * k_shared[i_s, i_k]
                T.reduce_sum(dg_last_fragment_2, dg_last_fragment_scalar_2, dim=-1, clear=False)
                dg_last_local_1 = dg_last_fragment_scalar_2[0]

                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    ds_fragment[i_s1, i_s2] = (
                        (ds_fragment[i_s1, i_s2] * T.exp(G[bb, bs * block_S + i_s1, bh] - G[bb, bs * block_S + i_s2, bh]) * scale)
                        if G[bb, bs * block_S + i_s1, bh] - G[bb, bs * block_S + i_s2, bh] <= 0
                        else 0
                    )

                T.clear(ds_fragment_positive)
                T.clear(ds_fragment_positive_transpose)
                T.gemm(q_shared, k_shared, ds_fragment_positive, transpose_B=True)
                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    ds_fragment_positive[i_s1, i_s2] = ds_fragment[i_s1, i_s2] * ds_fragment_positive[i_s1, i_s2]

                # FIXME: The reduce_sum statement with clear=True will cause an error of warp specialized pass
                T.reduce_sum(ds_fragment_positive, dg_fragment, dim=1, clear=False)
                T.copy(dg_fragment, dg_shared_1)

                # We should transpose the matrix because the reduce_sum statement can only reduce along the last dimension
                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    ds_fragment_positive_transpose[i_s2, i_s1] = ds_fragment_positive[i_s1, i_s2]

                # FIXME: The reduce_sum statement with clear=True will cause an error of warp specialized pass
                T.reduce_sum(ds_fragment_positive_transpose, dg_fragment_2, dim=1, clear=False)
                T.copy(dg_fragment_2, dg_shared_2)

                for i_s in T.Parallel(block_S):
                    dg_fragment_final[i_s] = dg_shared_1[i_s] - dg_shared_2[i_s]

                T.copy(ds_fragment, ds_shared)
                T.gemm(ds_shared, k_shared, dq_fragment)
                T.gemm(ds_shared, q_shared, dk_fragment, transpose_A=True)

                for i_s in T.Parallel(block_S):
                    dg_fragment_final[i_s] = dg_fragment_final[i_s] + dg_last_local_0 + dg_last_local_1

                T.copy(dq_fragment, dq[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK])
                T.copy(dk_fragment, dk[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK])
                for i_s in T.Parallel(block_S):
                    dg[bk, bb, bs * block_S + i_s, bh] = dg_fragment_final[i_s]

            else:
                for i_s1, i_s2 in T.Parallel(block_S, block_S):
                    ds_fragment[i_s1, i_s2] = 0 if i_s1 < i_s2 else ds_fragment[i_s1, i_s2]
                T.clear(dk_fragment_2)
                T.copy(ds_fragment, ds_shared)
                T.gemm(ds_shared, k_shared, dq_fragment)
                T.gemm(ds_shared, q_shared, dk_fragment_2, transpose_A=True)
                for i_s, i_k in T.Parallel(block_S, block_DK):
                    dq_fragment[i_s, i_k] = dq_fragment[i_s, i_k] * scale
                    dk_fragment[i_s, i_k] = dk_fragment[i_s, i_k] + dk_fragment_2[i_s, i_k] * scale
                T.copy(dq_fragment, dq[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK])
                T.copy(dk_fragment, dk[bb, bs * block_S : (bs + 1) * block_S, bh, bk * block_DK : (bk + 1) * block_DK])

    return kernel


def do_bench(fn, *args, warmup=10, rep=10, **kwargs):
    """
    Do benchmark for a function.
    """
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    for _ in range(warmup):
        fn(*args, **kwargs)

    torch.cuda.synchronize()
    for i in range(rep):
        start_event[i].record()
        fn(*args, **kwargs)
        end_event[i].record()
    torch.cuda.synchronize()

    # Record clocks
    times = torch.tensor(
        [s.elapsed_time(e) for s, e in zip(start_event, end_event)],
        dtype=torch.float,
    )

    return times.mean().item()


def run_test(
    B,
    S,
    H,
    DK,
    DV,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
    chunk_size,
    scale,
    use_g=True,
    use_dw=True,
    block_DK=64,
    block_DV=64,
    threads=256,
    num_stages=0,
):
    Q, K, V, h, G, dO, dh, dv, W = prepare_input(
        B,
        S,
        H,
        DK,
        DV,
        chunk_size,
        getattr(torch, input_dtype),
        getattr(torch, output_dtype),
        getattr(torch, accum_dtype),
        getattr(torch, gate_dtype),
        getattr(torch, state_dtype),
    )
    dq_ref, dk_ref, dw_ref, dg_ref = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype), block_DK
    )
    dq_tilelang, dk_tilelang, dw_tilelang, dg_tilelang = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype), block_DK
    )

    # ref
    if use_g:
        dq_ref, dk_ref, dw_ref, dg_ref = chunk_bwd_dqkwg(Q, K, V, G, dO, h, dh, dv, W, chunk_size=chunk_size, scale=scale)
    else:
        dq_ref, dk_ref, dw_ref, dg_ref = chunk_bwd_dqkwg(Q, K, V, None, dO, h, dh, dv, W, chunk_size=chunk_size, scale=scale)

    # tilelang
    kernel = tilelang_chunk_o_bwd_dqkwg(
        B,
        S,
        H,
        DK,
        DV,
        input_dtype,
        output_dtype,
        accum_dtype,
        gate_dtype,
        state_dtype,
        chunk_size,
        scale,
        use_g,
        use_dw,
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    dq_tilelang, dk_tilelang, dw_tilelang, dg_tilelang = kernel(Q, K, V, h, G, dO, dh, dv, W)

    if use_g:
        dg_tilelang = dg_tilelang.sum(dim=0)

    # check
    try:
        assert_similar(dq_ref, dq_tilelang, 1e-5, "tilelang chunk o bwd dq")
        print("tilelang chunk o bwd dq passed √")
    except Exception as e:
        print("tilelang chunk o bwd dq failed ✗")
        print(e)

    try:
        assert_similar(dk_ref, dk_tilelang, 1e-5, "tilelang chunk o bwd dk")
        print("tilelang chunk o bwd dk passed √")
    except Exception as e:
        print("tilelang chunk o bwd dk failed ✗")
        print(e)

    if use_g:
        try:
            assert_similar(dg_ref, dg_tilelang, 1e-5, "tilelang chunk o bwd dg")
            print("tilelang chunk o bwd dg passed √")
        except Exception as e:
            print("tilelang chunk o bwd dg failed ✗")
            print(e)

    if use_dw:
        try:
            assert_similar(dw_ref, dw_tilelang, 1e-5, "tilelang chunk o bwd dw")
            print("tilelang chunk o bwd dw passed √")
        except Exception as e:
            print("tilelang chunk o bwd dw failed ✗")
            print(e)


def main():
    DK = 128
    DV = 128
    run_test(
        B=1,
        S=32768,
        H=8,
        DK=DK,
        DV=DV,
        input_dtype=T.bfloat16,
        output_dtype=T.bfloat16,
        accum_dtype=T.float32,
        gate_dtype=T.float32,
        state_dtype=T.float32,
        chunk_size=64,
        scale=DK**-0.5,
        # scale=1,
        use_g=True,
        use_dw=True,
        block_DK=64,
        block_DV=64,
        threads=128,
        num_stages=0,
    )


if __name__ == "__main__":
    main()
