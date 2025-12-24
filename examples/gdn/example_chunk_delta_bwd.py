# Reference: fla/ops/common/chunk_delta_h.py

import sys  # noqa: F401

import tilelang
import tilelang.language as T

print(tilelang.__file__, flush=True)

# Add your fla repository path to sys.path
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae
# sys.path.insert(0, "/home/tzj/flash-linear-attention")
try:
    import fla

    print(fla.__file__, flush=True)
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
except ImportError:
    print("fla not found, using tilelang implementation")
    fla = None

import torch
import torch.nn.functional as F

torch.random.manual_seed(0)
# torch.set_printoptions(profile="full")

from test_utils import assert_similar


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
    Q = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    K = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    K = F.normalize(K, dim=-1, p=2)
    W = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    # Note: G should be in logspace and do chunkwise cumsum
    G = torch.randn(B, S, H, dtype=gate_dtype).cuda()
    G = F.logsigmoid(G)
    try:
        from fla.ops.utils.cumsum import chunk_local_cumsum

        G = chunk_local_cumsum(G, chunk_size)
    except ImportError:
        print("fla not found, skip cumsum")

    h0 = torch.randn(B, H, DK, DV, dtype=input_dtype).cuda()
    dht = torch.randn(B, H, DK, DV, dtype=input_dtype).cuda()
    dO = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    dv = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    return Q, K, W, G, h0, dht, dO, dv


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
    Q = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    K = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    W = torch.ones(B, S, H, DK, dtype=input_dtype).cuda()
    G = torch.ones(B, S, H, dtype=gate_dtype).cuda()
    h0 = torch.ones(B, H, DK, DV, dtype=input_dtype).cuda()
    dht = torch.ones(B, H, DK, DV, dtype=input_dtype).cuda()
    dO = torch.ones(B, S, H, DV, dtype=input_dtype).cuda()
    dv = torch.ones(B, S, H, DV, dtype=input_dtype).cuda()
    return Q, K, W, G, h0, dht, dO, dv


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
):
    BS = S // chunk_size
    dh = torch.empty(B, BS, H, DK, DV, dtype=output_dtype).cuda()
    dh0 = torch.empty(B, H, DK, DV, dtype=state_dtype).cuda()
    dv2 = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return dh, dh0, dv2


def torch_chunk_gated_delta_rule_bwd_dhu(
    Q: torch.Tensor,
    K: torch.Tensor,
    W: torch.Tensor,
    G: torch.Tensor,
    h0: torch.Tensor,
    dht: torch.Tensor,
    dO: torch.Tensor,
    dv: torch.Tensor,
    scale: float,
    use_g: bool,
    use_initial_state: bool,
    use_final_state_gradient: bool,
    input_dtype,
    output_dtype,
    accum_dtype,
    gate_dtype,
    state_dtype,
):
    B, S, H, DK = Q.shape
    DV = dv.shape[-1]
    block_S = 64
    BS = S // block_S
    dh, dh0, dv2 = (
        torch.empty((B, BS, H, DK, DV), dtype=output_dtype),
        torch.empty((B, H, DK, DV), dtype=state_dtype),
        torch.empty((B, S, H, DV), dtype=output_dtype),
    )
    dh_tmp = torch.empty((B, H, DK, DV), dtype=accum_dtype)
    dv_tmp = torch.empty((B, S, H, DV), dtype=accum_dtype)
    Q_tmp = torch.empty((B, S, H, DK), dtype=accum_dtype)

    if use_final_state_gradient:
        dh_tmp = dht.clone().to(accum_dtype)
    else:
        dh_tmp = torch.zeros_like(dht).to(accum_dtype)

    for i_s in range(BS - 1, -1, -1):
        dh[:, i_s, :, :, :] = dh_tmp
        dv_tmp = torch.matmul(K[:, i_s * block_S : (i_s + 1) * block_S, :, :].permute(0, 2, 1, 3), dh_tmp.to(K.dtype)).permute(0, 2, 1, 3)
        if use_g:
            for i_bh in range(B * H):
                i_b, i_h = i_bh // H, i_bh % H
                for i_s2 in range(block_S):
                    if G[i_b, i_s * block_S + block_S - 1, i_h] - G[i_b, i_s * block_S + i_s2, i_h] <= 0:
                        dv_tmp[i_b, i_s2, i_h, :] *= torch.exp(G[i_b, i_s * block_S + block_S - 1, i_h] - G[i_b, i_s * block_S + i_s2, i_h])
                    else:
                        dv_tmp[i_b, i_s2, i_h, :] = 0
        dv_tmp += dv[:, i_s * block_S : (i_s + 1) * block_S, :, :]
        dv2[:, i_s * block_S : (i_s + 1) * block_S, :, :] = dv_tmp

        if use_g:
            G_last = G[:, i_s * block_S + block_S - 1, :]
            for i_bh in range(B * H):
                i_b, i_h = i_bh // H, i_bh % H
                dh_tmp[i_b, i_h, :, :] *= torch.exp(G_last[i_b, i_h])
            Q_tmp = Q[:, i_s * block_S : (i_s + 1) * block_S, :, :]
            for i_s2 in range(block_S):
                for i_k in range(DK):
                    Q_tmp[:, i_s2, :, i_k] *= torch.exp(G[:, i_s * block_S + i_s2, :])
        Q_tmp *= scale
        W_tmp = W[:, i_s * block_S : (i_s + 1) * block_S, :, :]
        dO_tmp = dO[:, i_s * block_S : (i_s + 1) * block_S, :, :]

        torch.backends.cuda.matmul.allow_tf32 = True
        dh_tmp += torch.matmul(Q_tmp.permute(0, 2, 3, 1), dO_tmp.permute(0, 2, 1, 3))
        dh_tmp -= torch.matmul(W_tmp.permute(0, 2, 3, 1), dv_tmp.permute(0, 2, 1, 3))
        torch.backends.cuda.matmul.allow_tf32 = False

    if use_initial_state:
        dh0 = dh_tmp[:, :, :, :]
    else:
        dh0 = torch.zeros_like(dh_tmp[:, :, :, :])
    print(dh0.dtype)

    return dh, dh0, dv2


@tilelang.jit(out_idx=[-3, -2, -1])
def tilelang_chunk_gated_delta_rule_bwd_dhu(
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
    use_initial_state=True,
    use_final_state_gradient=True,
    # kernel config
    block_DV=64,
    threads=256,
    num_stages=0,
):
    block_S = chunk_size
    # Should support cu_seqlen
    BS = S // block_S

    Q_shape = (B, S, H, DK)
    K_shape = (B, S, H, DK)
    W_shape = (B, S, H, DK)
    G_shape = (B, S, H)
    h0_shape = (B, H, DK, DV)
    dht_shape = (B, H, DK, DV)
    dO_shape = (B, S, H, DV)
    dv_shape = (B, S, H, DV)

    dh_shape = (B, BS, H, DK, DV)
    dh0_shape = (B, H, DK, DV)
    dv2_shape = (B, S, H, DV)

    @T.prim_func
    def kernel(
        # Input
        Q: T.Tensor(Q_shape, dtype=input_dtype),
        K: T.Tensor(K_shape, dtype=input_dtype),
        W: T.Tensor(W_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        h0: T.Tensor(h0_shape, dtype=input_dtype),
        dht: T.Tensor(dht_shape, dtype=input_dtype),
        dO: T.Tensor(dO_shape, dtype=input_dtype),
        dv: T.Tensor(dv_shape, dtype=input_dtype),
        # Output
        dh: T.Tensor(dh_shape, dtype=output_dtype),
        dh0: T.Tensor(dh0_shape, dtype=state_dtype),
        dv2: T.Tensor(dv2_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            b_dh_shared = T.alloc_shared((DK, block_DV), dtype=output_dtype)
            b_dh_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            b_dh_fragment_1 = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            b_dh_fragment_2 = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            dv_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            dv_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dv_fragment_2 = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            dO_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            dO_shared_t = T.alloc_shared((block_DV, block_S), dtype=T.float32)
            dO_fragment = T.alloc_fragment((block_S, block_DV), dtype=T.float32)
            dO_fragment_t = T.alloc_fragment((block_DV, block_S), dtype=T.float32)
            K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)

            Q_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)

            G_shared = T.alloc_shared((block_S), dtype=gate_dtype, scope="shared")
            G_fragment = T.alloc_fragment((block_S), dtype=gate_dtype)
            G_fragment_post = T.alloc_fragment((block_S), dtype=gate_dtype)
            G_fragment_exp = T.alloc_fragment((block_S), dtype=gate_dtype)
            Q_fragment = T.alloc_fragment((block_S, DK), dtype=accum_dtype)
            Q_fragment_t = T.alloc_fragment((DK, block_S), dtype=accum_dtype)

            T.use_swizzle(10)

            T.annotate_layout(
                {
                    dO_shared: tilelang.layout.make_swizzled_layout(dO_shared),
                    Q_shared: tilelang.layout.make_swizzled_layout(Q_shared),
                }
            )

            if use_final_state_gradient:
                T.copy(dht[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], b_dh_shared)
                T.copy(b_dh_shared, b_dh_fragment)
            else:
                T.clear(b_dh_fragment)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # The gradient should be stored in the reverse order
                i_s_inv = T.ceildiv(S, block_S) - i_s - 1

                # Store the updated dh
                T.copy(b_dh_fragment, b_dh_shared)
                T.copy(b_dh_shared, dh[bb, i_s_inv, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

                # Update dv
                T.copy(K[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, 0:DK], K_shared)
                T.gemm(K_shared, b_dh_shared, dv_fragment, clear_accum=True)

                if use_g:
                    T.copy(G[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh], G_shared, disable_tma=True)
                    T.copy(G_shared, G_fragment)
                    G_last_local = G_shared[block_S - 1]
                    G_last_local_exp = T.exp(G_last_local)
                    for i_s2 in T.Parallel(block_S):
                        G_fragment_post[i_s2] = T.exp(G_last_local - G_fragment[i_s2])
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        dv_fragment[i_s2, i_v] = (
                            dv_fragment[i_s2, i_v] * G_fragment_post[i_s2] if G_last_local - G_fragment[i_s2] <= 0 else 0
                        )

                T.copy(dv[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV], dv_shared)
                T.copy(dv_shared, dv_fragment_2)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    dv_fragment[i_s2, i_v] = dv_fragment[i_s2, i_v] + dv_fragment_2[i_s2, i_v]

                # Store the updated dv
                T.copy(dv_fragment, dv_shared)
                T.copy(dv_shared, dv2[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV])

                # Update dh
                T.copy(Q[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, 0:DK], Q_shared)
                T.copy(W[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, 0:DK], W_shared)

                T.clear(Q_fragment)
                if use_g:
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_dh_fragment[i_k, i_v] *= G_last_local_exp
                    T.copy(Q_shared, Q_fragment)
                    for i_s2 in T.Parallel(block_S):
                        G_fragment_exp[i_s2] = T.exp(G_shared[i_s2])
                    for i_s2, i_k in T.Parallel(block_S, DK):
                        Q_fragment[i_s2, i_k] = Q_fragment[i_s2, i_k] * G_fragment_exp[i_s2] * scale
                else:
                    T.copy(Q_shared, Q_fragment)
                    for i_s2, i_k in T.Parallel(block_S, DK):
                        Q_fragment[i_s2, i_k] = Q_fragment[i_s2, i_k] * scale
                # Get transpose of Q_fragment to meet tf32 gemm requirement
                for i_s2, i_k in T.Parallel(block_S, DK):
                    Q_fragment_t[i_k, i_s2] = Q_fragment[i_s2, i_k]

                T.copy(dO[bb, i_s_inv * block_S : (i_s_inv + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV], dO_shared)
                T.copy(dO_shared, dO_fragment)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    dO_fragment_t[i_v, i_s2] = dO_fragment[i_s2, i_v]
                T.copy(dO_fragment_t, dO_shared_t)

                T.clear(b_dh_fragment_1)
                T.gemm(Q_fragment_t, dO_shared_t, b_dh_fragment_1, transpose_B=True)
                T.clear(b_dh_fragment_2)
                T.gemm(W_shared, dv_shared, b_dh_fragment_2, transpose_A=True)
                for i_k, i_v in T.Parallel(DK, block_DV):
                    b_dh_fragment[i_k, i_v] += b_dh_fragment_1[i_k, i_v] - b_dh_fragment_2[i_k, i_v]

            if use_initial_state:
                T.copy(b_dh_fragment, dh0[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

    return kernel


def test_result(dh_0, dh0_0, dv2_0, dh_1, dh0_1, dv2_1, name):
    try:
        torch.testing.assert_close(dh_0, dh_1, rtol=1e-2, atol=1e-2, equal_nan=True)
        print(f"{name} dh_0 and dh_1 passed for {name}")
    except Exception as e:
        print(f"{name} dh_0 and dh_1 are not close for {name}")
        print(e, end="\n\n")
    try:
        torch.testing.assert_close(dh0_0, dh0_1, rtol=1e-2, atol=1e-2, equal_nan=True)
        print(f"{name} dh0_0 and dh0_1 passed for {name}")
    except Exception as e:
        print(f"{name} dh0_0 and dh0_1 are not close for {name}")
        print(e, end="\n\n")
    try:
        torch.testing.assert_close(dv2_0, dv2_1, rtol=1e-2, atol=1e-2, equal_nan=True)
        print(f"{name} dv2_0 and dv2_1 passed for {name}")
    except Exception as e:
        print(f"{name} dv2_0 and dv2_1 are not close for {name}")
        print(e, end="\n\n")

    close = torch.isclose(dh_0, dh_1, rtol=1e-2, atol=1e-2)
    mismatch_indices = torch.nonzero(~close, as_tuple=True)
    error_num = 0
    for indices in zip(*mismatch_indices):
        if error_num < 100:
            print(
                f"{name} dh_0[{[idx.item() for idx in indices]}] = {dh_0[indices[0].item(), indices[1].item(), indices[2].item(), indices[3].item(), indices[4].item()]}, dh_1[{[idx.item() for idx in indices]}] = {dh_1[indices[0].item(), indices[1].item(), indices[2].item(), indices[3].item(), indices[4].item()]}"
            )
            error_num += 1
    close = torch.isclose(dh0_0, dh0_1, rtol=1e-2, atol=1e-2)
    mismatch_indices = torch.nonzero(~close, as_tuple=True)
    error_num = 0
    for indices in zip(*mismatch_indices):
        if error_num < 100:
            print(
                f"{name} dh0_0[{[idx.item() for idx in indices]}] = {dh0_0[indices[0].item(), indices[1].item(), indices[2].item(), indices[3].item()]}, dh0_1[{[idx.item() for idx in indices]}] = {dh0_1[indices[0].item(), indices[1].item(), indices[2].item(), indices[3].item()]}"
            )
            error_num += 1
    close = torch.isclose(dv2_0, dv2_1, rtol=1e-2, atol=1e-2)
    mismatch_indices = torch.nonzero(~close, as_tuple=True)
    error_num = 0
    for indices in zip(*mismatch_indices):
        if error_num < 100:
            print(
                f"{name} dv2_0[{[idx.item() for idx in indices]}] = {dv2_0[indices[0].item(), indices[1].item(), indices[2].item(), indices[3].item()]}, dv2_1[{[idx.item() for idx in indices]}] = {dv2_1[indices[0].item(), indices[1].item(), indices[2].item(), indices[3].item()]}"
            )
            error_num += 1


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
    use_initial_state=True,
    use_final_state_gradient=True,
    block_DV=64,
    threads=256,
    num_stages=0,
    use_torch=False,
):
    Q, K, W, G, h0, dht, dO, dv = prepare_input(
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
    dh_ref, dh0_ref, dv2_ref = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )
    dh_tilelang, dh0_tilelang, dv2_tilelang = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )

    # fla ref
    print("fla running...", flush=True)
    if use_g:
        dh_ref, dh0_ref, dv2_ref = chunk_gated_delta_rule_bwd_dhu(Q, K, W, G, h0, dht, dO, dv, scale)
    else:
        G = G.fill_(0)
        dh_ref, dh0_ref, dv2_ref = chunk_gated_delta_rule_bwd_dhu(Q, K, W, G, h0, dht, dO, dv, scale)

    # tilelang
    print("tilelang running...", flush=True)
    kernel = tilelang_chunk_gated_delta_rule_bwd_dhu(
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
        use_initial_state,
        use_final_state_gradient,
        block_DV,
        threads,
        num_stages,
    )
    # kernel = tilelang.compile(program)
    print(kernel.get_kernel_source())
    dh_tilelang, dh0_tilelang, dv2_tilelang = kernel(Q, K, W, G, h0, dht, dO, dv)

    fla_time = do_bench(chunk_gated_delta_rule_bwd_dhu, Q, K, W, G, h0, dht, dO, dv, scale, chunk_size=chunk_size)
    tilelang_time = do_bench(kernel, Q, K, W, G, h0, dht, dO, dv)

    print(f"fla time: {fla_time} ms")
    print(f"tilelang time: {tilelang_time} ms")

    assert_similar(dh_tilelang, dh_ref, 1e-5, "fla-tilelang", data="dh")
    assert_similar(dh0_tilelang, dh0_ref, 1e-5, "fla-tilelang", data="dh0")
    assert_similar(dv2_tilelang, dv2_ref, 1e-5, "fla-tilelang", data="dv2")

    # torch ref
    if use_torch:
        print("torch running...", flush=True)
        if use_g:
            dh_ref_torch, dh0_ref_torch, dv2_ref_torch = torch_chunk_gated_delta_rule_bwd_dhu(
                Q,
                K,
                W,
                G,
                h0,
                dht,
                dO,
                dv,
                scale,
                use_g,
                use_initial_state,
                use_final_state_gradient,
                getattr(torch, input_dtype),
                getattr(torch, output_dtype),
                getattr(torch, accum_dtype),
                getattr(torch, gate_dtype),
                getattr(torch, state_dtype),
            )
            dh_ref_torch = dh_ref_torch.cuda()
            dh0_ref_torch = dh0_ref_torch.cuda()
            dv2_ref_torch = dv2_ref_torch.cuda()
        else:
            dh_ref_torch, dh0_ref_torch, dv2_ref_torch = torch_chunk_gated_delta_rule_bwd_dhu(
                Q,
                K,
                W,
                None,
                h0,
                dht,
                dO,
                dv,
                scale,
                use_g,
                use_initial_state,
                use_final_state_gradient,
                getattr(torch, input_dtype),
                getattr(torch, output_dtype),
                getattr(torch, accum_dtype),
                getattr(torch, gate_dtype),
                getattr(torch, state_dtype),
            )
            dh_ref_torch = dh_ref_torch.cuda()
            dh0_ref_torch = dh0_ref_torch.cuda()
            dv2_ref_torch = dv2_ref_torch.cuda()

        assert_similar(dh_ref_torch, dh_ref, 1e-5, "torch-fla", data="dh")
        assert_similar(dh0_ref_torch, dh0_ref, 1e-5, "torch-fla", data="dh0")
        assert_similar(dv2_ref_torch, dv2_ref, 1e-5, "torch-fla", data="dv2")
        assert_similar(dh_ref_torch, dh_tilelang, 1e-5, "torch-tilelang", data="dh")
        assert_similar(dh0_ref_torch, dh0_tilelang, 1e-5, "torch-tilelang", data="dh0")
        assert_similar(dv2_ref_torch, dv2_tilelang, 1e-5, "torch-tilelang", data="dv2")


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


def main():
    DK = 128
    run_test(
        B=1,
        S=32768,
        H=8,
        DK=DK,
        DV=128,
        input_dtype=T.bfloat16,
        output_dtype=T.bfloat16,
        accum_dtype=T.float32,
        gate_dtype=T.float32,
        state_dtype=T.float32,
        chunk_size=64,
        scale=DK**-0.5,
        use_g=True,
        use_initial_state=True,
        use_final_state_gradient=True,
        block_DV=32,
        threads=128,
        num_stages=1,
        use_torch=False,
    )


if __name__ == "__main__":
    main()
