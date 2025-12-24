import torch
from tilelang import language as T

B = 1
S = 1024  # small but for test only.
H = 32
DK = 128
DV = 128
input_dtype = T.bfloat16
output_dtype = T.bfloat16
accum_dtype = T.float32
gate_dtype = T.float32
state_dtype = T.float32
chunk_size = 64
use_g = True
use_initial_state = True
store_final_state = True
use_final_state_gradient = True
save_new_value = True
block_DK = 64
block_DV = 32
threads = 128
num_stages = 0


def test_example_wy_fast_compilation():
    from example_wy_fast import tilelang_recompute_w_u_fwd, prepare_input

    K, V, Beta, G, A = prepare_input(
        B, S, H, DK, DV, chunk_size, getattr(torch, input_dtype), getattr(torch, output_dtype), gate_dtype=getattr(torch, gate_dtype)
    )
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


def test_example_wy_fast_bwd_split_compilation():
    from example_wy_fast_bwd_split import tilelang_wy_fast_bwd, tilelang_wy_fast_bwd_split, prepare_input, prepare_output

    K, V, Beta, G, A, dw, du = prepare_input(
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
    dk_tilelang, dv_tilelang, dbeta_tilelang, dg_tilelang = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, gate_dtype), getattr(torch, state_dtype)
    )
    BS = chunk_size
    dA_tilelang = torch.empty(B, S, H, BS, dtype=getattr(torch, input_dtype)).cuda()
    dbeta_tilelang_k = torch.empty(B, S, H, dtype=getattr(torch, output_dtype)).cuda()
    dg_tilelang_A_positive = torch.empty(B, S, H, BS, dtype=getattr(torch, gate_dtype)).cuda()
    dg_tilelang_A_negative = torch.empty(B, S, H, BS, dtype=getattr(torch, gate_dtype)).cuda()

    # tilelang
    kernel = tilelang_wy_fast_bwd(
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
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    dA_tilelang, dk_tilelang, dv_tilelang, dbeta_tilelang, dg_tilelang = kernel(K, V, Beta, G, A, dw, du)
    torch.cuda.synchronize()
    kernel_split = tilelang_wy_fast_bwd_split(
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
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    kernel_split(
        K, V, Beta, G, A, dw, du, dA_tilelang, dk_tilelang, dv_tilelang, dbeta_tilelang_k, dg_tilelang_A_positive, dg_tilelang_A_negative
    )
    torch.cuda.synchronize()

    dbeta_tilelang = dbeta_tilelang_k + dbeta_tilelang
    dg_tilelang = dg_tilelang + dg_tilelang_A_positive.sum(dim=-1) - dg_tilelang_A_negative.sum(dim=-1)


def test_example_chunk_o_compilation():
    from example_chunk_o import tilelang_chunk_fwd_o, prepare_input

    Q, K, V, HIDDEN, G = prepare_input(
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
    )
    scale = 1.0 / DK**0.5
    block_S = chunk_size
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
    O_tilelang = kernel(Q, K, V, HIDDEN, G)  # noqa: F841


def test_example_chunk_o_bwd_compilation():
    from example_chunk_o_bwd import tilelang_chunk_o_bwd_dqkwg, prepare_input

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
        1.0,
        use_g,
        True,
        block_DK,
        block_DV,
        threads,
        num_stages,
    )

    dq_tilelang, dk_tilelang, dw_tilelang, dg_tilelang = kernel(Q, K, V, h, G, dO, dh, dv, W)  # noqa: F841
    if use_g:
        dg_tilelang = dg_tilelang.sum(dim=0)


def test_example_chunk_scaled_dot_kkt_compilation():
    from example_chunk_scaled_dot_kkt import tilelang_chunk_scaled_dot_kkt_fwd, prepare_input

    K, Beta, G = prepare_input(B, S, H, DK, getattr(torch, input_dtype), getattr(torch, output_dtype), getattr(torch, accum_dtype))
    block_S = chunk_size
    kernel = tilelang_chunk_scaled_dot_kkt_fwd(
        B, S, H, DK, chunk_size, input_dtype, output_dtype, accum_dtype, use_g, block_S, block_DK, threads, num_stages
    )
    A_tilelang = kernel(K, Beta, G)  # noqa: F841


def test_example_cumsum_compilation():
    from example_cumsum import tilelang_chunk_local_cumsum_scalar, prepare_cumsum_input, prepare_cumsum_output

    G = prepare_cumsum_input(B, S, H, getattr(torch, gate_dtype))
    G_new_tilelang = prepare_cumsum_output(B, S, H, getattr(torch, gate_dtype))
    block_S = chunk_size
    kernel = tilelang_chunk_local_cumsum_scalar(
        B=B,
        S=S,
        H=H,
        chunk_size=chunk_size,
        reverse=False,
        head_first=False,
        input_dtype=gate_dtype,
        output_dtype=gate_dtype,
        block_S=block_S,
        threads=threads,
        use_fragment=False,
    )
    G_new_tilelang = kernel(G)  # noqa: F841


def test_example_chunk_delta_h_compilation():
    from example_chunk_delta_h import tilelang_chunk_gated_delta_rule_fwd_h, prepare_input

    K, W, U, G, initial_state = prepare_input(
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
    )
    kernel = tilelang_chunk_gated_delta_rule_fwd_h(
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
        use_g,
        use_initial_state,
        store_final_state,
        save_new_value,
        block_DK,
        block_DV,
        threads,
        num_stages,
    )
    h_tilelang, final_state_tilelang, V_new_tilelang = kernel(K, W, U, G, initial_state)  # noqa: F841


def test_example_chunk_delta_bwd_compilation():
    from example_chunk_delta_bwd import tilelang_chunk_gated_delta_rule_bwd_dhu, prepare_input

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
        1.0,
        use_g,
        use_initial_state,
        use_final_state_gradient,
        block_DV,
        threads,
        num_stages,
    )
    dh_tilelang, dh0_tilelang, dv2_tilelang = kernel(Q, K, W, G, h0, dht, dO, dv)  # noqa: F841


if __name__ == "__main__":
    # tilelang.testing.main()
    test_example_chunk_delta_bwd_compilation()
