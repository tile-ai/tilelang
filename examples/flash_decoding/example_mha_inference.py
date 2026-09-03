import torch
import torch.nn.functional as F
import tilelang
from tilelang.autotuner import *
from tilelang.carver.arch import get_arch
import tilelang.language as T
from functools import partial

num_split = 4
_KERNEL_THREADS = 128
_TILE_CANDIDATES = ((128, 64), (64, 64))


@tilelang.jit(out_idx=[3])
def flashattn(batch, heads, seqlen_q, seqlen_kv, dim, is_causal, block_M, block_N):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    shape_q = [batch, seqlen_q, heads, dim]
    shape_kv = [batch, seqlen_kv, heads, dim]
    part_shape = [batch, seqlen_q, heads, num_split, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def flashattn_mha_inference(
        Q: T.Tensor(shape_q, dtype),
        K: T.Tensor(shape_kv, dtype),
        V: T.Tensor(shape_kv, dtype),
        Output: T.Tensor(shape_q, dtype),
    ):
        glse = T.alloc_global([batch, heads, num_split, seqlen_q], dtype)
        Output_partial = T.alloc_global(part_shape, dtype)  # [batch, seqlen_q, heads, num_split, dim]
        # split
        with T.Kernel(T.ceildiv(seqlen_q, block_M), heads * batch, num_split, threads=_KERNEL_THREADS) as (bx, by, bz):
            Q_shared = T.alloc_shared([block_M, dim], dtype)
            K_shared = T.alloc_shared([block_N, dim], dtype)
            V_shared = T.alloc_shared([block_N, dim], dtype)
            O_shared = T.alloc_shared([block_M, dim], dtype)
            acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
            acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
            scores_max = T.alloc_fragment([block_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
            scores_scale = T.alloc_fragment([block_M], accum_dtype)
            scores_sum = T.alloc_fragment([block_M], accum_dtype)
            logsum = T.alloc_fragment([block_M], accum_dtype)

            mid = bx
            hid = by % heads
            bid = by // heads
            sid = bz

            # NOTE(wt): tma barrier has some problems with padded dimensions (seq_q here) currently
            # disable relevant tma copy and use SIMT as fallback for now
            T.copy(Q[bid, mid * block_M : (mid + 1) * block_M, hid, :], Q_shared, disable_tma=True)
            T.fill(acc_o, 0)
            T.fill(logsum, 0)
            T.fill(scores_max, -T.infinity(accum_dtype))

            # TODO: Handle causal split case
            loop_range = (
                T.min(T.ceildiv(seqlen_kv, block_N), T.ceildiv((mid + 1) * block_M, block_N))
                if is_causal
                else T.ceildiv((seqlen_kv // num_split), block_N)
            )

            for k in T.Pipelined(loop_range, num_stages=2):
                T.copy(
                    K[bid, (seqlen_kv // num_split) * sid + k * block_N : (seqlen_kv // num_split) * sid + (k + 1) * block_N, hid, :],
                    K_shared,
                )
                # TODO: Handle causal split case
                if is_causal:
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.if_then_else(mid * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                else:
                    T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                # To do causal softmax, we need to set the scores_max to 0 if it is -inf
                # This process is called Check_inf in FlashAttention3 code, and it only need to be done
                # in the first ceil_div(kBlockM, kBlockN) steps.
                # for i in T.Parallel(block_M):
                #     scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
                for i in T.Parallel(block_M):
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, block_N):
                    # Instead of computing exp(x - max), we compute exp2(x * log_2(e) -
                    # max * log_2(e)) This allows the compiler to use the ffma
                    # instruction instead of fadd and fmul separately.
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]

                T.copy(
                    V[bid, (seqlen_kv // num_split) * sid + k * block_N : (seqlen_kv // num_split) * sid + (k + 1) * block_N, hid, :],
                    V_shared,
                )
                T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            for i in T.Parallel(block_M):
                logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
            T.copy(logsum, glse[bid, hid, sid, mid * block_M : (mid + 1) * block_M])
            T.copy(acc_o, O_shared)
            T.copy(O_shared, Output_partial[bid, mid * block_M : (mid + 1) * block_M, hid, sid, :], disable_tma=True)

        # combine
        with T.Kernel(T.ceildiv(seqlen_q, block_M), heads, batch, threads=_KERNEL_THREADS) as (bx, by, bz):
            po_local = T.alloc_fragment([block_M, dim], dtype)
            o_accum_local = T.alloc_fragment([block_M, dim], accum_dtype)
            o_shared = T.alloc_shared([block_M, dim], dtype)
            lse_local = T.alloc_fragment([num_split, block_M], dtype)
            lse_local_split = T.alloc_fragment([block_M], accum_dtype)
            lse_logsum_local = T.alloc_fragment([block_M], accum_dtype)
            lse_max_local = T.alloc_fragment([block_M], accum_dtype)
            scale_local = T.alloc_fragment([block_M], accum_dtype)

            T.clear(lse_logsum_local)
            T.clear(o_accum_local)
            T.copy(
                glse[
                    bz,
                    by,
                    :,
                    bx * block_M : (bx + 1) * block_M,
                ],
                lse_local,
            )
            T.reduce_max(lse_local, lse_max_local, dim=0, clear=True)
            for k in T.Pipelined(num_split):
                T.copy(lse_local[k, :], lse_local_split)
                for i in T.Parallel(block_M):
                    lse_logsum_local[i] += T.exp2(lse_local_split[i] - lse_max_local[i])
            for i in T.Parallel(block_M):
                lse_logsum_local[i] = T.log2(lse_logsum_local[i]) + lse_max_local[i]
            for k in T.Pipelined(num_split, num_stages=2):
                T.copy(Output_partial[bz, bx * block_M : (bx + 1) * block_M, by, k, :], po_local)
                for i in T.Parallel(block_M):
                    lse_local_split[i] = lse_local[k, i]
                for i in T.Parallel(block_M):
                    scale_local[i] = T.exp2(lse_local_split[i] - lse_logsum_local[i])
                for i, j in T.Parallel(block_M, dim):
                    o_accum_local[i, j] += po_local[i, j] * scale_local[i]
            T.copy(o_accum_local, o_shared)
            T.copy(o_shared, Output[bz, bx * block_M : (bx + 1) * block_M, by, :], disable_tma=True)

    return flashattn_mha_inference


def _max_dynamic_shared_memory(kernel):
    """Return the largest dynamic shared-memory allocation in a compiled module."""
    allocations = getattr(kernel.adapter, "dynamic_smem_buf", {}).values()
    return max((int(size) for size in allocations if size is not None), default=0)


def _fits_current_hip_device(kernel):
    """Check generated HIP resources before attempting a potentially invalid launch."""
    arch = get_arch(kernel.target)
    dynamic_smem = _max_dynamic_shared_memory(kernel)
    if dynamic_smem > arch.smem_cap:
        return False, f"dynamic shared memory {dynamic_smem} exceeds device limit {arch.smem_cap}"

    for name, usage in kernel.resource_usage.items():
        if usage.n_regs and usage.n_regs * _KERNEL_THREADS >= arch.reg_cap:
            return False, f"{name} uses {usage.n_regs} VGPRs per thread at the device register-file limit"

    return True, ""


def compile_flashattn_for_current_device(batch, heads, seqlen_q, seqlen_kv, dim, is_causal):
    """Compile the largest feasible MHA tile for the current device.

    CUDA keeps the original 128x64 configuration. HIP candidates are checked
    against the generated module's actual LDS and VGPR usage before launch so
    devices with larger resource budgets can retain the larger tile.
    """
    rejected = []
    for block_M, block_N in _TILE_CANDIDATES:
        kernel = flashattn(batch, heads, seqlen_q, seqlen_kv, dim, is_causal, block_M, block_N)
        config = {"block_M": block_M, "block_N": block_N}
        if kernel.target.kind.name != "hip":
            return kernel, config

        fits, reason = _fits_current_hip_device(kernel)
        if fits:
            return kernel, config
        rejected.append(f"{block_M}x{block_N}: {reason}")

    details = "; ".join(rejected)
    raise RuntimeError(f"No feasible flash-decoding tile for the current HIP device ({details}).")


def ref_program(Q, K, V, causal):
    assert causal is False
    dim = Q.size(-1)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V)
    return output


def reduce_ref(Q, K, V, glse, Output_partial, causal):
    o = torch.empty_like(Output_partial[:, :, :, 0, :]).fill_(0)
    lse_logsum = torch.empty_like(glse[:, :, 0, :]).fill_(0)  # [batch, seqlen_q, heads]
    lse_max = glse.max(dim=2, keepdim=False).values
    for ks in range(num_split):
        lse = glse[:, :, ks, :]
        lse_logsum += torch.exp2(lse - lse_max)
    lse_logsum = torch.log2(lse_logsum) + lse_max
    for ks in range(num_split):
        lse = glse[:, :, ks, :]
        scale = torch.exp2(lse - lse_logsum)  # [batch, heads, seqlen_q]
        o += Output_partial[:, :, :, ks, :] * scale[:, :, :, None].transpose(1, 2)
    return o.to(torch.float16)


def flash_split_ref(Q, K, V, causal):
    # [batch, seqlen_q, heads, dim]
    batch = Q.size(0)
    block_M = Q.size(1)
    nheads = Q.size(2)
    dim = Q.size(3)
    block_N = 128
    seqlen_kv = K.size(1)

    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e)
    acc_s = torch.empty((batch, nheads, block_M, block_N), device="cuda", dtype=torch.float)
    acc_s_cast = torch.empty((batch, nheads, block_M, block_N), device="cuda", dtype=torch.float16)
    acc_o = torch.empty((batch, block_M, nheads, dim), device="cuda", dtype=torch.float)
    scores_max = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_max_prev = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_scale = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    scores_sum = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    logsum = torch.empty((batch, nheads, block_M), device="cuda", dtype=torch.float)
    gacc_o = torch.empty((num_split, batch, block_M, nheads, dim), device="cuda", dtype=torch.float)
    glogsum = torch.empty((num_split, batch, nheads, block_M), device="cuda", dtype=torch.float)

    Q_ = Q * scale

    for ks in range(num_split):
        acc_o.fill_(0)
        logsum.fill_(0)
        scores_max.fill_(float("-inf"))
        scores_max_prev.fill_(float("-inf"))
        for i in range(int((seqlen_kv // num_split) / block_N)):
            acc_s.fill_(0)
            acc_s = torch.einsum(
                "bqhd,bkhd->bhqk",
                Q_,
                K[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :],
            )  # [batch, seqlen, nheads, block_N]
            scores_max_prev = scores_max
            scores_max = acc_s.max(dim=-1, keepdim=False).values  # [blockM]
            scores_scale = torch.exp2(scores_max_prev - scores_max)
            acc_o *= scores_scale[:, :, :, None].transpose(1, 2)
            acc_s = torch.exp2(acc_s - scores_max[:, :, :, None])
            acc_s_cast = acc_s.to(torch.float16)
            acc_o += torch.einsum(
                "bhqk,bkhd->bqhd",
                acc_s_cast,
                V[:, (seqlen_kv // num_split) * ks + i * block_N : (seqlen_kv // num_split) * ks + (i + 1) * block_N, :, :],
            )
            scores_sum = acc_s.sum(dim=-1, keepdim=False)
            logsum = logsum * scores_scale + scores_sum
        acc_o /= logsum[:, :, :, None].transpose(1, 2)
        logsum = torch.log2(logsum) + scores_max
        gacc_o[ks, :, :, :, :] = acc_o
        glogsum[ks, :, :, :] = logsum

    return glogsum.to(torch.float16).permute(1, 2, 0, 3), gacc_o.to(torch.float16).permute(1, 2, 3, 0, 4)


def main(BATCH=1, H=32, Q_CTX=128, KV_CTX=8192, D_HEAD=128, causal=False):
    flops_per_matmul = 2.0 * BATCH * H * Q_CTX * KV_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    kernel, config = compile_flashattn_for_current_device(BATCH, H, Q_CTX, KV_CTX, D_HEAD, causal)
    print(f"Selected tile: {config['block_M']}x{config['block_N']}")
    ref_fn = partial(ref_program, causal=causal)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    profiler.assert_allclose(ref_fn, rtol=0.01, atol=0.01)
    print("All checks passed!")

    latency = profiler.do_bench(ref_fn, warmup=500)
    print("{:.2f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))
    latency = profiler.do_bench(n_warmup=10, n_repeat=10)
    print("{:.4f} ms".format(latency))
    print("{:.2f} TFlops".format(total_flops / latency * 1e-9))


def run_regression_perf(BATCH=1, H=32, Q_CTX=128, KV_CTX=8192, D_HEAD=128, causal=False):
    BLOCK_M = 128
    BLOCK_N = 64
    kernel = flashattn(BATCH, H, Q_CTX, KV_CTX, D_HEAD, causal, BLOCK_M, BLOCK_N)
    profiler = kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main()
