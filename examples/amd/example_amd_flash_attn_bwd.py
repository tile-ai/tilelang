import argparse
import hashlib
import itertools
import math
import os
import re
import subprocess
import tempfile
import time
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

import tilelang
import tilelang.language as T
from tilelang import env as tilelang_env
from tilelang.primitives.gemm.base import GemmWarpPolicy


def ref_program(Q, K, V, is_causal, groups=1):
    assert Q.size(2) == K.size(2) * groups, f"Q heads {Q.size(2)} K heads {K.size(2)} groups {groups}"
    assert Q.size(2) == V.size(2) * groups, f"Q heads {Q.size(2)} V heads {V.size(2)} groups {groups}"
    dim = Q.size(-1)
    K_ref = K.repeat_interleave(groups, dim=2)
    V_ref = V.repeat_interleave(groups, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", Q, K_ref)
    scores = scores / torch.sqrt(torch.tensor(dim, dtype=scores.dtype))
    if is_causal:
        seq_len = Q.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=scores.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.einsum("bhqk,bkhd->bqhd", attention_weights, V_ref)
    lse = torch.logsumexp(scores, dim=-1).float()
    return output, lse


def get_fwd_configs():
    block_M = [32, 64, 128, 256]
    block_N = [32, 64, 128, 256]
    threads = [128, 256, 512]
    num_split_q = [64, 128, 256]
    num_stages = [0, 1]
    enable_rasterization = [True]
    k_pack = [2]
    panel_size = [7, 8, 9, 10]
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


def _sanitize_filename(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]", "_", name)


def _get_hip_include_options() -> list[str]:
    options: list[str] = []
    template_path = tilelang_env.TILELANG_TEMPLATE_PATH
    if template_path:
        options.append(f"-I{template_path}")
    ck_path = tilelang_env.COMPOSABLE_KERNEL_INCLUDE_DIR
    if ck_path:
        options.append(f"-I{ck_path}")
    return options


def _compile_hip_to_asm(code: str, asm_path: str, arch: str | None = None, options: list[str] | None = None) -> None:
    from tilelang.contrib import hipcc  # pylint: disable=import-outside-toplevel

    if arch is None:
        rocm_path = hipcc.find_rocm_path()
        arch = hipcc.get_rocm_arch(rocm_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_code = os.path.join(temp_dir, "kernel.cc")
        with open(temp_code, "w", encoding="utf-8") as f:
            f.write(code)

        cmd = ["hipcc", "-O3", "-c"]
        if arch:
            cmd.append(f"--offload-arch={arch}")
        cmd.append("--genco")
        cmd.append("-S")
        include_options = _get_hip_include_options()
        if options:
            if isinstance(options, str):
                include_options.append(options)
            else:
                include_options.extend(options)
        if include_options:
            cmd.extend(include_options)
        cmd.extend(["-o", asm_path, temp_code])

        proc = subprocess.run(
            cmd,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"hipcc assembly generation failed:\n{proc.stdout}")


def dump_kernel_artifacts(kernel, dump_dir: str | None, tag: str) -> None:
    if not dump_dir or kernel is None:
        return

    try:
        kernel_source = kernel.get_kernel_source()
    except AttributeError:
        print(f"[TileLang] Kernel `{tag}` does not expose source; skip dumping.")
        return

    if not kernel_source:
        print(f"[TileLang] Kernel `{tag}` returned empty source; skip dumping.")
        return

    os.makedirs(dump_dir, exist_ok=True)

    prim_func = getattr(kernel, "prim_func", None)
    func_name = "kernel"
    if prim_func is not None and hasattr(prim_func, "attrs"):
        func_name = prim_func.attrs.get("global_symbol", func_name)

    digest = hashlib.sha1(kernel_source.encode("utf-8")).hexdigest()[:8]
    base = _sanitize_filename(f"{tag}_{func_name}_{digest}")

    hip_path = os.path.join(dump_dir, f"{base}.hip.cpp")
    with open(hip_path, "w", encoding="utf-8") as f:
        f.write(kernel_source)
    print(f"[TileLang] HIP source saved to {hip_path}")

    from tilelang.contrib import hipcc  # pylint: disable=import-outside-toplevel

    hsaco_path = os.path.join(dump_dir, f"{base}.hsaco")
    hip_options = _get_hip_include_options()
    try:
        hipcc.compile_hip(
            kernel_source,
            target_format="hsaco",
            options=hip_options if hip_options else None,
            path_target=hsaco_path,
        )
        print(f"[TileLang] HSACO saved to {hsaco_path}")
    except Exception as err:  # noqa: BLE001
        print(f"[TileLang] HIP compilation failed for {hip_path}: {err}")
        return

    asm_path = os.path.join(dump_dir, f"{base}.s")
    try:
        _compile_hip_to_asm(kernel_source, asm_path)
        print(f"[TileLang] Assembly saved to {asm_path}")
    except Exception as err:  # noqa: BLE001
        print(f"[TileLang] Assembly generation failed for {hip_path}: {err}")


@tilelang.autotune(configs=get_fwd_configs(), cache_input_tensors=True)
@tilelang.jit(out_idx=[3, 4], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
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
        LSE: T.Tensor([batch, heads, seq_len], accum_dtype),
    ):
        with T.Kernel(num_split_q, batch * heads, threads=threads) as (b_split, byz_combined):
            T.use_swizzle(panel_size, enable=enable_rasterization)

            bz = byz_combined // heads
            by = byz_combined % heads

            num_q_blocks = T.ceildiv(seq_len, block_M)

            bx_loop_var = T.alloc_var(T.int32)
            bx_loop_var = b_split

            with T.While(bx_loop_var < num_q_blocks):
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                m_i = T.alloc_fragment([block_M], accum_dtype)
                l_i = T.alloc_fragment([block_M], accum_dtype)

                T.fill(acc_o, 0)
                T.fill(m_i, -T.infinity(accum_dtype))
                T.fill(l_i, 0)

                current_bx = bx_loop_var
                q_block_offset = current_bx * block_M

                # Forward: Q在register里, K/V在LDS里
                Q_fragment = T.alloc_fragment([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)

                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                m_prev = T.alloc_fragment([block_M], accum_dtype)
                scale_factor = T.alloc_fragment([block_M], accum_dtype)

                T.copy(Q[bz, q_block_offset : q_block_offset + block_M, by, :], Q_fragment, coalesced_width=vec_size)

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
                        Q_fragment,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        k_pack=k_pack,
                        policy=GemmWarpPolicy.FullRow,
                    )

                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = acc_s[i, j] * scale

                    T.copy(m_i, m_prev)
                    T.reduce_max(acc_s, m_i, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        m_i[i] = T.max(m_i[i], m_prev[i])

                    for i in T.Parallel(block_M):
                        if m_prev[i] == -T.infinity(accum_dtype):
                            scale_factor[i] = 0.0
                        else:
                            scale_factor[i] = T.exp(m_prev[i] - m_i[i])

                        l_i[i] *= scale_factor[i]

                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scale_factor[i]

                    for i, j in T.Parallel(block_M, block_N):
                        if acc_s[i, j] == -T.infinity(acc_s.dtype):
                            acc_s[i, j] = 0.0
                        else:
                            acc_s[i, j] = T.exp(acc_s[i, j] - m_i[i])

                    T.reduce_sum(acc_s, row_sum, dim=1)
                    for i in T.Parallel(block_M):
                        l_i[i] += row_sum[i]

                    T.copy(acc_s, acc_s_cast)

                    T.gemm(acc_s_cast, V_shared, acc_o, policy=GemmWarpPolicy.FullRow)

                l_inv = T.alloc_fragment([block_M], accum_dtype)
                for i in T.Parallel(block_M):
                    safe_l = T.if_then_else(l_i[i] > 1e-6, l_i[i], 1.0)
                    l_inv[i] = 1.0 / safe_l

                for i, j in T.Parallel(block_M, dim):
                    Output[bz, q_block_offset + i, by, j] = acc_o[i, j] * l_inv[i]

                for i in T.Parallel(block_M):
                    if q_block_offset + i < seq_len:
                        lse_val = T.if_then_else(l_i[i] > 0, T.log(l_i[i]) + m_i[i], -T.infinity(accum_dtype))
                        LSE[bz, by, q_block_offset + i] = lse_val

                bx_loop_var = current_bx + num_split_q

    return main


def get_bwd_configs():
    block_M = [16, 32, 64, 128, 256]
    block_N = [16, 32, 64, 128, 256]
    threads = [64, 128, 256, 512, 1024]
    num_stages = [0, 1, 2]
    enable_rasterization = [True]
    panel_size = [7, 8, 9, 10]

    configs = []
    for m, n, stages, t, r, p in itertools.product(block_M, block_N, num_stages, threads, enable_rasterization, panel_size):
        configs.append(
            {
                "block_M": m,
                "block_N": n,
                "num_stages": stages,
                "threads": t,
                "enable_rasterization": r,
                "panel_size": p,
            }
        )

    return configs


@tilelang.jit(out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_bwd_preprocess(batch, heads, seq_len, dim):
    dtype = T.float16
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 32

    @T.prim_func
    def flash_bwd_prep(O: T.Tensor(shape, dtype), dO: T.Tensor(shape, dtype), Delta: T.Tensor([batch, heads, seq_len], accum_dtype)):
        with T.Kernel(batch, heads, T.ceildiv(seq_len, blk)) as (bz, bx, by):
            o = T.alloc_fragment([blk, blk], dtype)
            do = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(O[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], o)
                T.copy(dO[bz, by * blk : (by + 1) * blk, bx, k * blk : (k + 1) * blk], do)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o[i, j] * do[i, j]
            T.reduce_sum(acc, delta, 1)
            T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

    return flash_bwd_prep


@tilelang.autotune(configs=get_bwd_configs(), cache_input_tensors=True)
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_bwd(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    groups,
    block_M: int,
    block_N: int,
    num_stages: int,
    threads: int,
    enable_rasterization: bool,
    panel_size: int,
):
    sm_scale = (1.0 / dim) ** 0.5
    head_kv = heads // groups
    q_shape = [batch, seq_len, heads, dim]
    kv_shape = [batch, seq_len, head_kv, dim]
    dtype = T.float16
    accum_dtype = T.float32

    @T.prim_func
    def flash_bwd_kernel(
        Q: T.Tensor(q_shape, dtype),
        K: T.Tensor(kv_shape, dtype),
        V: T.Tensor(kv_shape, dtype),
        dO: T.Tensor(q_shape, dtype),
        lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        dQ_partial: T.Tensor([batch, T.ceildiv(seq_len, block_M), seq_len, heads, dim], dtype),
        dK: T.Tensor(kv_shape, accum_dtype),
        dV: T.Tensor(kv_shape, accum_dtype),
    ):
        with T.Kernel(head_kv, T.ceildiv(seq_len, block_M), batch, threads=threads) as (bk, by, bz):
            T.use_swizzle(panel_size, enable=enable_rasterization)

            # Backward: K在shared里, V在register里, Q/QT/dO/dOT在LDS里
            K_shared = T.alloc_shared([block_M, dim], dtype)
            V_fragment = T.alloc_fragment([block_M, dim], dtype)
            q_shared = T.alloc_shared([block_N, dim], dtype)
            do_shared = T.alloc_shared([block_N, dim], dtype)
            lse_shared = T.alloc_shared([block_N], accum_dtype)
            delta_shared = T.alloc_shared([block_N], accum_dtype)
            ds_shared = T.alloc_shared([block_M, block_N], dtype)

            qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
            P_acc = T.alloc_fragment([block_M, block_N], accum_dtype)
            dP = T.alloc_fragment([block_M, block_N], accum_dtype)

            dv = T.alloc_fragment([block_M, dim], accum_dtype)
            dk = T.alloc_fragment([block_M, dim], accum_dtype)
            dq = T.alloc_fragment([block_N, dim], accum_dtype)

            kv_offset = by * block_M

            T.copy(K[bz, kv_offset : kv_offset + block_M, bk, :], K_shared)
            T.copy(V[bz, kv_offset : kv_offset + block_M, bk, :], V_fragment)
            T.clear(dv)
            T.clear(dk)

            loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
            loop_ed = T.ceildiv(seq_len, block_N)

            for group_idx in range(groups):
                q_head = bk * groups + group_idx
                if q_head >= heads:
                    continue

                for k in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(Q[bz, k * block_N : (k + 1) * block_N, q_head, :], q_shared)
                    T.clear(qkT)

                    T.gemm(K_shared, q_shared, qkT, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(lse[bz, q_head, k * block_N : (k + 1) * block_N], lse_shared)

                    for i, j in T.Parallel(block_M, block_N):
                        P_acc[i, j] = T.exp(qkT[i, j] * sm_scale - lse_shared[j])

                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            P_acc[i, j] = T.if_then_else(kv_offset + i <= k * block_N + j, P_acc[i, j], 0.0)

                    T.copy(dO[bz, k * block_N : (k + 1) * block_N, q_head, :], do_shared)
                    T.copy(Delta[bz, q_head, k * block_N : (k + 1) * block_N], delta_shared)

                    # Optimized GEMM order: Compute dV first (GEMM1), then dP (GEMM2)
                    # This improves V_fragment reuse since it's already in registers
                    # GEMM1: P^T @ dO = dV (compute dV first)
                    P_cast = T.alloc_fragment([block_M, block_N], dtype)
                    for i, j in T.Parallel(block_M, block_N):
                        P_cast[i, j] = T.cast(P_acc[i, j], dtype)
                    T.gemm(P_cast, do_shared, dv, policy=T.GemmWarpPolicy.FullRow)

                    # GEMM2: dO @ V^T = dP (compute dP after dV, reusing V_fragment in registers)
                    T.clear(dP)
                    T.gemm(V_fragment, do_shared, dP, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

                    # Compute ds = P * (dP - delta) * sm_scale for dk and dq computation
                    # Create ds_compute fragment directly in loop to avoid layout conflict
                    ds_compute = T.alloc_fragment([block_M, block_N], dtype)
                    for i, j in T.Parallel(block_M, block_N):
                        ds_compute[i, j] = T.cast(P_acc[i, j] * (dP[i, j] - delta_shared[j]) * sm_scale, dtype)

                    # Compute dK: ds^T @ Q = dK
                    T.gemm(ds_compute, q_shared, dk, policy=T.GemmWarpPolicy.FullRow)

                    # Compute dQ: ds @ K = dQ (using shared memory for K)
                    T.copy(ds_compute, ds_shared)
                    T.clear(dq)
                    T.gemm(ds_shared, K_shared, dq, transpose_A=True)
                    for i, j in T.Parallel(block_N, dim):
                        seq_idx = k * block_N + i
                        if seq_idx < seq_len and q_head < heads:
                            dQ_partial[bz, by, seq_idx, q_head, j] = T.cast(dq[i, j], dtype)

            for i, j in T.Parallel(block_M, dim):
                seq_idx = kv_offset + i
                if seq_idx < seq_len:
                    dV[bz, seq_idx, bk, j] = dv[i, j]
                    dK[bz, seq_idx, bk, j] = dk[i, j]

    return flash_bwd_kernel


@tilelang.jit(out_idx=[1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_bwd_postprocess(batch, heads, seq_len, dim):
    dtype = T.float16
    accum_dtype = T.float32
    shape = [batch, seq_len, heads, dim]
    blk = 64

    @T.prim_func
    def flash_bwd_post(dQ_in: T.Tensor(shape, accum_dtype), dQ_out: T.Tensor(shape, dtype)):
        with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (bx, by, bz):
            T.copy(
                dQ_in[bz, bx * blk : (bx + 1) * blk, by, :],
                dQ_out[bz, bx * blk : (bx + 1) * blk, by, :],
            )

    return flash_bwd_post


@tilelang.jit(out_idx=[1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def flashattn_bwd_reduce_dq(batch, num_tiles, seq_len, heads, dim):
    accum_dtype = "float"

    @T.prim_func
    def flash_bwd_reduce(
        dQ_partial: T.Tensor([batch, num_tiles, seq_len, heads, dim], "float16"),
        dQ_out: T.Tensor([batch, seq_len, heads, dim], accum_dtype),
    ):
        with T.Kernel(batch, heads, seq_len, threads=64) as (bz, bh, bs):
            acc = T.alloc_fragment([dim], accum_dtype)
            T.clear(acc)

            for tile in range(num_tiles):
                for d in T.Parallel(dim):
                    val = T.cast(dQ_partial[bz, tile, bs, bh, d], accum_dtype)
                    acc[d] += val

            for d in T.Parallel(dim):
                dQ_out[bz, bs, bh, d] = acc[d]

    return flash_bwd_reduce


def debug_tensor_comparison(tensor1, tensor2, name, rtol=1e-3, atol=1e-3):
    print(f"\n=== {name} Comparison ===")
    print(f"Shape: {tensor1.shape} vs {tensor2.shape}")
    print(f"Data type: {tensor1.dtype} vs {tensor2.dtype}")
    print(f"Device: {tensor1.device} vs {tensor2.device}")

    diff = torch.abs(tensor1 - tensor2)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    std_diff = diff.std().item()

    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Difference std: {std_diff:.6f}")

    if max_diff > atol:
        max_idx = torch.argmax(diff)
        max_idx = np.unravel_index(max_idx.cpu().numpy(), tensor1.shape)
        print(f"Max difference position: {max_idx}")
        print(f"Value1: {tensor1[max_idx].item():.6f}, Value2: {tensor2[max_idx].item():.6f}")

    nan_count1 = torch.isnan(tensor1).sum().item()
    nan_count2 = torch.isnan(tensor2).sum().item()
    inf_count1 = torch.isinf(tensor1).sum().item()
    inf_count2 = torch.isinf(tensor2).sum().item()

    print(f"NaN count: {nan_count1} vs {nan_count2}")
    print(f"Inf count: {inf_count1} vs {inf_count2}")

    relative_diff = diff / (torch.abs(tensor2) + 1e-8)
    max_relative_diff = relative_diff.max().item()
    mean_relative_diff = relative_diff.mean().item()

    print(f"Max relative difference: {max_relative_diff:.6f}")
    print(f"Mean relative difference: {mean_relative_diff:.6f}")

    close = torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    print(f"Within tolerance (rtol={rtol}, atol={atol}): {close}")

    return close, max_diff, mean_diff


def benchmark_function(func, *args, warmup=10, repeat=100):
    for _ in range(warmup):
        func(*args)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = time.time()
        func(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        times.append((end - start) * 1000)

    return np.median(times)


def main(
    batch: int = 1,
    heads: int = 8,
    seq_len: int = 4096,
    dim: int = 128,
    is_causal: bool = False,
    groups: int = 1,
    dump_dir: str | None = None,
):
    device = "cuda"
    dtype = torch.float16

    if not dump_dir:
        dump_dir = os.path.join(os.getcwd(), "tilelang_kernel_dumps")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    print(f"Test configuration: batch={batch}, heads={heads}, seq_len={seq_len}, dim={dim}, is_causal={is_causal}, groups={groups}")

    flops_per_gemm = 2.0 * batch * heads * seq_len * seq_len * dim
    total_flops = 5 * flops_per_gemm

    print(f"Total FLOPs: {total_flops / 1e12:.2f} TFlops")

    q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq_len, heads // groups, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq_len, heads // groups, dim, device=device, dtype=dtype)
    dO = torch.randn_like(q)

    print("Starting autotuning for Fast FlashAttention-V2 Forward Pass...")
    fwd_kernel = fast_flashattn(batch, heads, seq_len, dim, is_causal, groups)
    if fwd_kernel is None or fwd_kernel.config is None:
        print("Forward pass auto-tuning failed.")
        return
    print(f"Autotuning finished. Best Forward Configuration: {fwd_kernel.config}")

    ref_program_processed = partial(ref_program, is_causal=is_causal, groups=groups)

    profiler = fwd_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    print("Verifying correctness...")
    profiler.assert_allclose(ref_program_processed, rtol=0.01, atol=0.01)
    print("Forward pass is correct.")

    dump_kernel_artifacts(fwd_kernel, dump_dir, "flashattn_fwd")

    o_tl, lse_tl = fwd_kernel(q, k, v)

    bwd_prep = flashattn_bwd_preprocess(batch, heads, seq_len, dim)
    dump_kernel_artifacts(bwd_prep, dump_dir, "flashattn_bwd_preprocess")
    delta_tl = bwd_prep(o_tl, dO)

    print("\nStarting FlashAttention-V2 backward pass autotuning...")
    bwd_kernel = flashattn_bwd(batch, heads, seq_len, dim, is_causal, groups)
    if bwd_kernel is None or bwd_kernel.config is None:
        print("Backward pass autotuning failed.")
        return
    print(f"Autotuning completed. Best backward pass configuration: {bwd_kernel.config}")

    dump_kernel_artifacts(bwd_kernel, dump_dir, "flashattn_bwd")

    block_M = bwd_kernel.config["block_M"]
    num_kv_tiles = math.ceil(seq_len / block_M)

    dQ_partial = torch.zeros((batch, num_kv_tiles, seq_len, heads, dim), dtype=torch.float16, device=device)
    dK_tl = torch.zeros_like(k, dtype=torch.float32)
    dV_tl = torch.zeros_like(v, dtype=torch.float32)

    bwd_kernel(q, k, v, dO, lse_tl, delta_tl, dQ_partial, dK_tl, dV_tl)

    reduce_kernel = flashattn_bwd_reduce_dq(batch, num_kv_tiles, seq_len, heads, dim)
    dump_kernel_artifacts(reduce_kernel, dump_dir, "flashattn_bwd_reduce_dq")
    dQ_accum = reduce_kernel(dQ_partial)

    post_kernel = flashattn_bwd_postprocess(batch, heads, seq_len, dim)
    dump_kernel_artifacts(post_kernel, dump_dir, "flashattn_bwd_postprocess")
    dQ_tl = post_kernel(dQ_accum)

    q_ref = q.clone().detach().requires_grad_()
    k_ref = k.clone().detach().requires_grad_()
    v_ref = v.clone().detach().requires_grad_()

    o_ref, _ = ref_program(q_ref, k_ref, v_ref, is_causal, groups)
    o_ref.backward(dO)

    print("Verifying backward pass correctness...")
    dq_close, dq_max_diff, dq_mean_diff = debug_tensor_comparison(dQ_tl, q_ref.grad, "dQ", rtol=0.05, atol=0.05)
    if dq_close:
        print("dQ is correct.")
    else:
        print("dQ mismatch detected.")

    dk_close, dk_max_diff, dk_mean_diff = debug_tensor_comparison(dK_tl.to(torch.float16), k_ref.grad, "dK", rtol=0.05, atol=0.05)
    if dk_close:
        print("dK is correct.")
    else:
        print("dK mismatch detected.")

    dv_close, dv_max_diff, dv_mean_diff = debug_tensor_comparison(dV_tl.to(torch.float16), v_ref.grad, "dV", rtol=0.05, atol=0.05)
    if dv_close:
        print("dV is correct.")
    else:
        print("dV mismatch detected.")

    print("\n=== Performance Benchmarking ===")

    def run_reference_fwd_bwd():
        q_ref_bench = q.clone().detach().requires_grad_()
        k_ref_bench = k.clone().detach().requires_grad_()
        v_ref_bench = v.clone().detach().requires_grad_()

        o_ref_bench, _ = ref_program(q_ref_bench, k_ref_bench, v_ref_bench, is_causal, groups)

        o_ref_bench.backward(dO)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    ref_latency = benchmark_function(run_reference_fwd_bwd, warmup=10, repeat=100)
    print(f"Reference PyTorch Forward+Backward: {ref_latency:.2f} ms | {total_flops / ref_latency * 1e-9:.2f} TFlops")

    def run_complete_fwd_bwd():
        o_tl_bench, lse_tl_bench = fwd_kernel(q, k, v)

        delta_tl_bench = bwd_prep(o_tl_bench, dO)

        dQ_partial_bench = torch.zeros((batch, num_kv_tiles, seq_len, heads, dim), dtype=torch.float16, device=device)
        dK_bench = torch.zeros_like(k, dtype=torch.float32)
        dV_bench = torch.zeros_like(v, dtype=torch.float32)
        bwd_kernel(q, k, v, dO, lse_tl_bench, delta_tl_bench, dQ_partial_bench, dK_bench, dV_bench)
        dQ_bench = reduce_kernel(dQ_partial_bench)

        post_kernel(dQ_bench)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    tile_latency = benchmark_function(run_complete_fwd_bwd, warmup=10, repeat=100)
    print(
        f"Complete Flash Attention V2 Forward+Backward (Tile-lang): {tile_latency:.2f} ms | {total_flops / tile_latency * 1e-9:.2f} TFlops"
    )

    speedup = ref_latency / tile_latency
    print(f"Speedup: {speedup:.2f}x")

    print("Forward output: Passed")
    print(f"dQ: {'Passed' if dq_close else 'Failed'} (Max diff: {dq_max_diff:.6f})")
    print(f"dK: {'Passed' if dk_close else 'Failed'} (Max diff: {dk_max_diff:.6f})")
    print(f"dV: {'Passed' if dv_close else 'Failed'} (Max diff: {dv_max_diff:.6f})")

    if all([dq_close, dk_close, dv_close]):
        print("All checks passed!")
    else:
        print("Some checks failed, may need further debugging.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4, help="batch size")
    parser.add_argument("--heads", type=int, default=16, help="heads")
    parser.add_argument("--seq_len", type=int, default=4096, help="sequence length")
    parser.add_argument("--dim", type=int, default=64, help="dim")
    parser.add_argument("--is_causal", action="store_true", help="causal")
    parser.add_argument("--groups", type=int, default=1, help="groups")
    parser.add_argument("--dump_dir", type=str, default=None, help="directory to save HIP, HSACO and disassembled files")
    args = parser.parse_args()

    main(args.batch, args.heads, args.seq_len, args.dim, args.is_causal, args.groups, args.dump_dir)
