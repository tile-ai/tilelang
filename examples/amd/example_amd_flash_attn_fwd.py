import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy
import itertools
import argparse
from functools import partial


# Custom supply function to ensure tensors are created on GPU
def supply_tensors_gpu(params):
    """Supply function that creates tensors on GPU for ROCm/HIP."""
    tensors = []
    for param in params:
        if hasattr(param, "shape") and hasattr(param, "dtype"):
            # Force creation on GPU device
            shape = [int(s) for s in param.shape]
            tensor = torch.randn(shape, dtype=param.dtype, device="cuda")
            tensors.append(tensor)
        else:
            tensors.append(param)
    return tensors


def ref_program(Q, K, V, is_causal, groups=1):
    assert Q.size(2) == K.size(2) * groups, f"Q heads {Q.size(2)} K heads {K.size(2)} groups {groups}"
    assert Q.size(2) == V.size(2) * groups, f"Q heads {Q.size(2)} V heads {V.size(2)} groups {groups}"
    dim = Q.size(-1)
    K = K.repeat_interleave(groups, dim=2)
    V = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
    return output


def get_configs():
    """Generates configurations for the autotuner, tailored for FA-2 style parallelism."""
    block_M = [32, 64, 128, 256]
    block_N = [32, 64, 128, 256]
    threads = [128, 256, 512]
    num_split_q = [64, 128, 256]
    num_stages = [0, 1]
    enable_rasterization = [True]
    k_pack = [2]
    panel_size = [7, 8]
    qk_coalesced_width = [8]
    v_coalesced_width = [4]

    valid_configs = []

    for m, n, s, t, stages, r, k, p, qkw, vw in itertools.product(
        block_M, block_N, num_split_q, threads, num_stages, enable_rasterization, k_pack, panel_size, qk_coalesced_width, v_coalesced_width
    ):
        valid_configs.append(
            {
                "block_M": m,
                "block_N": n,
                "num_split_q": s,
                "threads": t,
                "num_stages": stages,
                "enable_rasterization": r,
                "k_pack": k,
                "panel_size": p,
                "qk_coalesced_width": qkw,
                "v_coalesced_width": vw,
            }
        )
    return valid_configs


@tilelang.autotune(configs=get_configs(), cache_input_tensors=True, supply_prog=supply_tensors_gpu)
@tilelang.jit(out_idx=[3])
def fast_flashattn(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups,
    block_M: int,
    block_N: int,
    num_split_q: int,
    threads: int,
    num_stages: int,
    enable_rasterization: bool,
    k_pack: int,
    panel_size: int,
    qk_coalesced_width: int,
    v_coalesced_width: int,
):
    scale = (1.0 / dim) ** 0.5
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    vec_size = qk_coalesced_width
    v_vec_size = v_coalesced_width

    @T.prim_func
    def main(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        Output: T.Tensor(q_shape, dtype),
    ):
        with T.Kernel(num_split_q, batch * heads, threads=threads) as (b_split, byz_combined):
            T.use_swizzle(panel_size, enable=enable_rasterization)

            bz = byz_combined // heads
            by = byz_combined % heads

            num_q_blocks = T.ceildiv(seq_len, block_M)

            bx = T.alloc_var(T.int32)
            bx = b_split

            with T.While(bx < num_q_blocks):
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                m_i = T.alloc_fragment([block_M], accum_dtype)
                l_i = T.alloc_fragment([block_M], accum_dtype)
                T.fill(acc_o, 0)
                T.fill(m_i, -T.infinity(accum_dtype))
                T.fill(l_i, 0)

                current_bx = bx
                q_block_offset = current_bx * block_M

                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                # Use register fragment for P instead of shared memory to reduce LDS usage
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)

                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                m_prev = T.alloc_fragment([block_M], accum_dtype)
                scale_factor = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, q_block_offset : q_block_offset + block_M, by, :], Q_shared, coalesced_width=vec_size)

                loop_end_k = T.ceildiv(q_block_offset + block_M, block_N) if is_causal else T.ceildiv(seq_len, block_N)

                row_sum = T.alloc_fragment([block_M], accum_dtype)

                for k in T.Pipelined(loop_end_k, num_stages=num_stages):
                    kv_idx = k * block_N

                    T.copy(K[bz, kv_idx : kv_idx + block_N, by // groups, :], K_shared, coalesced_width=vec_size)
                    T.copy(V[bz, kv_idx : kv_idx + block_N, by // groups, :], V_shared, coalesced_width=v_vec_size)

                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(q_block_offset + i >= kv_idx + j, 0, -T.infinity(acc_s.dtype))
                    else:
                        T.clear(acc_s)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        k_pack=k_pack,
                        policy=GemmWarpPolicy.FullRow,
                    )

                    T.copy(m_i, m_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        m_i[i] = T.max(m_i[i], m_prev[i])

                    for i in T.Parallel(block_M):
                        sf = T.exp(m_prev[i] * scale - m_i[i] * scale)
                        l_i[i] *= sf
                        scale_factor[i] = sf

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scale_factor[i]

                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp(acc_s[i, j] * scale - m_i[i] * scale)

                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for i in T.Parallel(block_M):
                        l_i[i] += row_sum[i]

                    # Cast acc_s (accum_dtype) to dtype in registers and directly GEMM with V
                    T.copy(acc_s, acc_s_cast)

                    T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.FullRow)

                l_inv = T.alloc_fragment([block_M], accum_dtype)
                for i in T.Parallel(block_M):
                    safe_l = T.if_then_else(l_i[i] > 1e-6, l_i[i], 1.0)
                    l_inv[i] = 1.0 / safe_l

                for i, j in T.Parallel(block_M, dim):
                    Output[bz, q_block_offset + i, by, j] = acc_o[i, j] * l_inv[i]

                bx = current_bx + num_split_q

    return main


def main(batch: int = 1, heads: int = 8, seq_len: int = 4096, dim: int = 128, is_causal: bool = False, groups: int = 1):
    flops_per_matmul = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 2 * flops_per_matmul
    if is_causal:
        total_flops *= 0.5

    print("Starting autotuning for FlashAttention-V2...")
    kernel = fast_flashattn(batch, heads, seq_len, dim, is_causal, groups=groups)
    print(f"Autotuning finished. Best Configuration: {kernel.config}")

    ref_program_processed = partial(ref_program, is_causal=is_causal, groups=groups)

    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    print("Verifying correctness...")
    profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
    print("All checks pass.")

    latency = profiler.do_bench(ref_program_processed, warmup=100)
    print(f"Reference (PyTorch): {latency:.2f} ms | {total_flops / latency * 1e-9:.2f} TFlops")

    latency = profiler.do_bench(warmup=100)
    print(f"Fast Flash Attention V2 (Tile-lang): {latency:.2f} ms | {total_flops / latency * 1e-9:.2f} TFlops")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1, help="batch size")
    parser.add_argument("--heads", type=int, default=8, help="heads")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--dim", type=int, default=128, help="dim")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument("--groups", type=int, default=1, help="groups")
    args = parser.parse_args()
    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.groups)
