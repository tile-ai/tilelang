# Reference: fla/ops/common/chunk_delta_h.py

import sys  # noqa: F401
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune

# Add your fla repository path to sys.path
# Currently we use the fla repository from the flash-linear-attention project at commit id f03cb3ae
# sys.path.insert(0, "/home/tzj/flash-linear-attention")
try:
    import fla

    print(fla.__file__)
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
except ImportError:
    print("fla not found, using tilelang implementation")
    fla = None

import torch
import torch.nn.functional as F
from tilelang.engine.callback import register_cuda_postproc_callback  # noqa: F401

from test_utils import assert_similar

# (zhengju) We can slightly modify the generated cuda code from tilelang lowering
# in the debug folder to make the performance better. To enable this callback,
# you can comment out the following function.
# @register_cuda_postproc_callback
# def tilelang_callback_cuda_postproc(code, _):
#     cuda_code = open("../debug/chunk_delta_h_fuse.cu", "r").read()
#     code = cuda_code
#     return code

torch.random.manual_seed(0)


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
    K = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    K = F.normalize(K, dim=-1, p=2)
    W = torch.randn(B, S, H, DK, dtype=input_dtype).cuda()
    W = F.normalize(W, dim=-1, p=2)
    U = torch.randn(B, S, H, DV, dtype=input_dtype).cuda()
    U = F.normalize(U, dim=-1, p=2)
    G = torch.randn(B, S, H, dtype=gate_dtype).cuda()
    G = F.logsigmoid(G)
    try:
        from fla.ops.utils.cumsum import chunk_local_cumsum

        G = chunk_local_cumsum(G, chunk_size)
    except ImportError:
        print("fla not found, skip cumsum")

    initial_state = torch.randn(B, H, DK, DV, dtype=input_dtype).cuda()
    return K, W, U, G, initial_state


def prepare_output(
    B,
    S,
    H,
    DK,
    DV,
    chunk_size,
    output_dtype,
    state_dtype,
):
    BS = S // chunk_size
    h = torch.empty(B, BS, H, DK, DV, dtype=output_dtype).cuda()
    final_state = torch.empty(B, H, DK, DV, dtype=state_dtype).cuda()
    V_new = torch.empty(B, S, H, DV, dtype=output_dtype).cuda()
    return h, final_state, V_new


def get_configs():
    import itertools

    block_DK = [32, 64, 128]
    block_DV = [32, 64, 128]
    threads = [128, 256]
    num_stages = [1, 2, 3]
    _configs = list(itertools.product(block_DK, block_DV, threads, num_stages))

    configs = [{"block_DK": c[0], "block_DV": c[1], "threads": c[2], "num_stages": c[3]} for c in _configs]
    return configs


@autotune(configs=get_configs(), warmup=3, rep=5)
@tilelang.jit(out_idx=[-3, -2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_chunk_gated_delta_rule_fwd_h(
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
    use_g,
    use_initial_state,
    store_final_state,
    save_new_value,
    # kernel config
    block_DK=64,
    block_DV=32,
    threads=128,
    num_stages=1,
):
    block_S = chunk_size
    BS = S // block_S

    K_shape = (B, S, H, DK)
    V_shape = (B, S, H, DV)
    W_shape = (B, S, H, DK)
    U_shape = (B, S, H, DV)
    G_shape = (B, S, H)
    h_shape = (B, BS, H, DK, DV)
    initial_state_shape = (B, H, DK, DV)
    final_state_shape = (B, H, DK, DV)

    @T.prim_func
    def kernel(
        K: T.Tensor(K_shape, dtype=input_dtype),
        W: T.Tensor(W_shape, dtype=input_dtype),
        U: T.Tensor(U_shape, dtype=input_dtype),
        G: T.Tensor(G_shape, dtype=gate_dtype),
        initial_state: T.Tensor(initial_state_shape, dtype=input_dtype),
        h: T.Tensor(h_shape, dtype=output_dtype),
        final_state: T.Tensor(final_state_shape, dtype=state_dtype),
        V_new: T.Tensor(V_shape, dtype=output_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), B * H, threads=threads) as (bv, bbh):
            bb, bh = bbh // H, bbh % H

            b_h_shared = T.alloc_shared((DK, block_DV), dtype=input_dtype)
            b_h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)

            U_shared = T.alloc_shared((block_S, block_DV), dtype=input_dtype)
            U_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            W_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            V_new_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            V_new_shared = T.alloc_shared((block_S, block_DV), dtype=output_dtype)
            K_shared = T.alloc_shared((block_S, DK), dtype=input_dtype)
            G_last_local = T.alloc_var(T.float32)
            G_shared = T.alloc_shared((block_S, block_DV), dtype=gate_dtype)
            G_fragment = T.alloc_fragment((block_S, block_DV), dtype=gate_dtype)

            T.annotate_layout(
                {
                    U_shared: tilelang.layout.make_swizzled_layout(U_shared),
                    G_shared: tilelang.layout.make_swizzled_layout(G_shared),
                }
            )

            T.use_swizzle(10)

            if use_initial_state:
                T.copy(initial_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV], b_h_shared)
                T.copy(b_h_shared, b_h_fragment)
            else:
                T.clear(b_h_fragment)

            for i_s in T.Pipelined(T.ceildiv(S, block_S), num_stages=num_stages):
                # Store previous result to the hidden tensor, like the epilogue
                T.copy(b_h_shared, h[bb, i_s, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

                # Recurrence
                T.copy(W[bb, i_s * block_S : (i_s + 1) * block_S, bh, 0:DK], W_shared)
                T.gemm(W_shared, b_h_shared, V_new_fragment, clear_accum=True)

                # U - W * S
                T.copy(U[bb, i_s * block_S : (i_s + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV], U_shared)
                T.copy(U_shared, U_fragment)
                for i_s2, i_v in T.Parallel(block_S, block_DV):
                    V_new_fragment[i_s2, i_v] = -V_new_fragment[i_s2, i_v] + U_fragment[i_s2, i_v]

                # Save V_new
                if save_new_value:
                    T.copy(V_new_fragment, dst=V_new_shared)
                    T.copy(V_new_shared, V_new[bb, i_s * block_S : (i_s + 1) * block_S, bh, bv * block_DV : (bv + 1) * block_DV])

                T.copy(K[bb, i_s * block_S : (i_s + 1) * block_S, bh, 0:DK], K_shared)
                # use_g
                if use_g:
                    G_last_local = G[bb, (i_s + 1) * block_S - 1, bh]
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        G_shared[i_s2, i_v] = G[bb, i_s * block_S + i_s2, bh]
                    T.copy(G_shared, G_fragment)
                    for i_s2, i_v in T.Parallel(block_S, block_DV):
                        V_new_fragment[i_s2, i_v] = (
                            V_new_fragment[i_s2, i_v] * T.exp2((G_last_local - G_fragment[i_s2, i_v]) * 1.442695)
                            if G_last_local - G_fragment[i_s2, i_v] <= 0
                            else 0
                        )
                    G_last_local = T.exp2(G_last_local * 1.442695)
                    for i_k, i_v in T.Parallel(DK, block_DV):
                        b_h_fragment[i_k, i_v] *= G_last_local

                # Update intermediate results
                T.copy(V_new_fragment, V_new_shared)
                T.gemm(K_shared, V_new_shared, b_h_fragment, transpose_A=True)

                T.copy(b_h_fragment, b_h_shared)

            # Save final state
            if store_final_state:
                T.copy(b_h_fragment, final_state[bb, bh, 0:DK, bv * block_DV : (bv + 1) * block_DV])

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
    use_g=True,
    use_initial_state=True,
    store_final_state=True,
    save_new_value=True,
    block_DK=64,
    block_DV=32,
    threads=128,
    num_stages=0,
):
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
    h_ref, final_state_ref, V_new_ref = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, state_dtype)
    )
    h_tilelang, final_state_tilelang, V_new_tilelang = prepare_output(
        B, S, H, DK, DV, chunk_size, getattr(torch, output_dtype), getattr(torch, state_dtype)
    )

    # fla ref
    h_ref, V_new_ref, final_state_ref = chunk_gated_delta_rule_fwd_h(
        k=K,
        w=W,
        u=U,
        g=G,
        initial_state=initial_state,
        output_final_state=store_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
    )

    # tilelang
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
    )
    h_tilelang, final_state_tilelang, V_new_tilelang = kernel(K, W, U, G, initial_state)
    # (zhengju) If you want to print the generated cuda code, you can uncomment the following line
    # print("CUDA Code:\n", kernel.get_kernel_source())

    fla_time = do_bench(
        chunk_gated_delta_rule_fwd_h,
        k=K,
        w=W,
        u=U,
        g=G,
        initial_state=initial_state,
        output_final_state=store_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
    )
    tilelang_time = do_bench(kernel, K, W, U, G, initial_state)

    # check correctness
    try:
        h_ref_fp32 = h_ref.to(torch.float32)
        h_tilelang_fp32 = h_tilelang.to(torch.float32)
        assert_similar(h_ref_fp32, h_tilelang_fp32, eps=1e-5, name="tilelang chunk gated delta rule fwd h", raise_assert=False)
        print("tilelang chunk gated delta rule fwd h passed √")
    except Exception as e:
        print("tilelang chunk gated delta rule fwd h failed ✗")
        print(e)

    try:
        final_state_ref_fp32 = final_state_ref.to(torch.float32)
        final_state_tilelang_fp32 = final_state_tilelang.to(torch.float32)
        assert_similar(
            final_state_ref_fp32,
            final_state_tilelang_fp32,
            eps=1e-5,
            name="tilelang chunk gated delta rule fwd final_state",
            raise_assert=False,
        )
        print("tilelang chunk gated delta rule fwd final_state passed √")
    except Exception as e:
        print("tilelang chunk gated delta rule fwd final_state failed ✗")
        print(e)

    try:
        V_new_ref_fp32 = V_new_ref.to(torch.float32)
        V_new_tilelang_fp32 = V_new_tilelang.to(torch.float32)
        assert_similar(V_new_ref_fp32, V_new_tilelang_fp32, eps=1e-5, name="tilelang chunk gated delta rule fwd V_new", raise_assert=False)
        print("tilelang chunk gated delta rule fwd V_new passed √")
    except Exception as e:
        print("tilelang chunk gated delta rule fwd V_new failed ✗")
        print(e)

    print(f"tilelang time: {tilelang_time} ms")
    print(f"fla time: {fla_time} ms")


def main():
    run_test(
        B=1,
        S=32768,
        H=32,
        DK=128,
        DV=128,
        input_dtype=T.bfloat16,
        output_dtype=T.bfloat16,
        accum_dtype=T.float32,
        gate_dtype=T.float32,
        state_dtype=T.float32,
        chunk_size=64,
        use_g=True,
        use_initial_state=False,
        store_final_state=True,
        save_new_value=True,
        block_DK=32,
        block_DV=32,
        threads=128,
        num_stages=2,
    )


if __name__ == "__main__":
    main()
