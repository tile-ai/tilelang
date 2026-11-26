"""TileLang Ascend implementation of the Engram gate forward/backward operators.

The implementation follows the mathematical definition of the CUDA operator,
but uses Ascend's GM -> UB vector execution model.  Forward uses SimdVF while
backward is deliberately split into three SimtVF kernels so every kernel has a
small, regular UB working set.
"""

import tilelang
from tilelang import language as T
from tilelang.language import simd as S


_VEC = 64  # fp32 lanes in one 2048-bit Ascend vector register


@tilelang.jit
def get_engram_gate_fwd_kernel_asc(
    hidden_size: int,
    eps: float,
    scalar: float,
    k_stride_s: int,
    k_stride_h: int,
    v_stride_s: int,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
    num_cores: int = 64,
    save_for_backward: bool = True,
):
    """Build the Ascend forward kernel."""
    num_tokens = T.dynamic("num_tokens")
    assert hidden_size % _VEC == 0, "Ascend SIMD Engram gate requires hidden_size % 64 == 0"
    assert num_cores % hc_mult == 0, "num_cores must be divisible by hc_mult"
    num_vectors = hidden_size // _VEC
    heads_per_task = 2 if hidden_size <= 6144 else 1
    num_head_groups = hc_mult // heads_per_task
    # Deeper buffering hides MTE latency for shapes that leave enough UB room.
    if heads_per_task == 2:
        num_stages = 6 if hidden_size <= 2560 else (5 if hidden_size <= 3072 else (3 if hidden_size <= 4096 else 2))
        v_stages = num_stages
    else:
        num_stages = 6 if hidden_size <= 3072 else (5 if hidden_size <= 4096 else (2 if hidden_size < 6144 else 3))
        v_stages = 2 if hidden_size >= 6144 else num_stages

    @T.prim_func
    def kernel(
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        k: T.StridedTensor[
            (num_tokens, hc_mult, hidden_size),
            (k_stride_s, k_stride_h, 1),
            T.bfloat16,
        ],
        v: T.StridedTensor[
            (num_tokens, hidden_size),
            (v_stride_s, 1),
            T.bfloat16,
        ],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float32],
        output: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        dot_out: T.Tensor[(num_tokens, hc_mult), T.float32],
        gate_out: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_x_out: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_k_out: T.Tensor[(num_tokens, hc_mult), T.float32],
    ) -> None:
        total_tasks = num_tokens * num_head_groups
        iterations = T.ceildiv(total_tasks, num_cores)

        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((heads_per_task, hidden_size), T.bfloat16)
            k_ub = T.alloc_shared((heads_per_task, hidden_size), T.bfloat16)
            v_ub = T.alloc_shared((hidden_size,), T.bfloat16)
            w_ub = T.alloc_shared((heads_per_task, hidden_size), T.float32)
            out_ub = T.alloc_shared((heads_per_task, hidden_size), T.bfloat16)
            dot_ub = T.alloc_shared((8,), T.float32)
            gate_ub = T.alloc_shared((8,), T.float32)
            rstd_x_ub = T.alloc_shared((8,), T.float32)
            rstd_k_ub = T.alloc_shared((8,), T.float32)
            T.annotate_buffer_versions(
                {
                    x_ub: num_stages,
                    k_ub: num_stages,
                    v_ub: v_stages,
                    out_ub: num_stages,
                    dot_ub: num_stages,
                    gate_ub: num_stages,
                    rstd_x_ub: num_stages,
                    rstd_k_ub: num_stages,
                }
            )

            # Persistent cores keep one adjacent head group and its weights in
            # UB. Grouped heads share the token's v load and scalar DMAs.
            head_group = core_id % num_head_groups
            head_base = head_group * heads_per_task
            T.copy(
                weight_fused[head_base : head_base + heads_per_task, :],
                w_ub[:heads_per_task, :hidden_size],
            )

            for task_iter in T.Pipelined(
                iterations,
                num_stages=num_stages,
                annotations={
                    "multi_buffer_eligible": [
                        x_ub,
                        k_ub,
                        v_ub,
                        out_ub,
                        dot_ub,
                        gate_ub,
                        rstd_x_ub,
                        rstd_k_ub,
                    ]
                },
            ):
                task = task_iter * num_cores + core_id
                if task < total_tasks:
                    token = task // num_head_groups

                    T.copy(
                        hidden_states[token, head_base : head_base + heads_per_task, :],
                        x_ub[:heads_per_task, :hidden_size],
                    )
                    T.copy(
                        k[token, head_base : head_base + heads_per_task, :],
                        k_ub[:heads_per_task, :hidden_size],
                    )
                    T.copy(v[token, :], v_ub[:hidden_size])

                    with T.SimdVF():
                        for local_head in T.unroll(heads_per_task, explicit=True):
                            full = S.pset(32, "PAT_ALL")
                            one_lane = S.pset(32, "PAT_VL1")
                            zero = S.vdup(0.0, T.float32, full)
                            one = S.vdup(1.0, T.float32, full)
                            sum_x2 = S.alloc_var(T.float32)
                            sum_k2 = S.alloc_var(T.float32)
                            raw_dot = S.alloc_var(T.float32)
                            sum_x2 = zero
                            sum_k2 = zero
                            raw_dot = zero

                            for vector in range(num_vectors):
                                offset = vector * _VEC
                                x_vec = S.vcvt(
                                    S.vld(x_ub[local_head, offset], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                k_vec = S.vcvt(
                                    S.vld(k_ub[local_head, offset], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                w_vec = S.vld(w_ub[local_head, offset])
                                S.vmula(sum_x2, x_vec, x_vec, full)
                                S.vmula(sum_k2, k_vec, k_vec, full)
                                raw_term = S.vmul(S.vmul(x_vec, w_vec), k_vec)
                                raw_dot = S.vadd(raw_dot, raw_term)

                            sum_x2 = S.vdupv(S.vcadd(sum_x2), full)
                            sum_k2 = S.vdupv(S.vcadd(sum_k2), full)
                            raw_dot = S.vdupv(S.vcadd(raw_dot), full)
                            rstd_x = S.vdiv(
                                one,
                                S.vsqrt(S.vadds(S.vmuls(sum_x2, 1.0 / hidden_size), eps)),
                            )
                            rstd_k = S.vdiv(
                                one,
                                S.vsqrt(S.vadds(S.vmuls(sum_k2, 1.0 / hidden_size), eps)),
                            )
                            normalized_dot = S.vmuls(S.vmul(S.vmul(raw_dot, rstd_x), rstd_k), scalar)
                            sqrt_abs = S.vsqrt(S.vmaxs(S.vabs(normalized_dot), clamp_value))
                            positive_sqrt = S.vsel(
                                sqrt_abs,
                                zero,
                                S.vcmps(normalized_dot, 0.0, full, "gt"),
                            )
                            signed_sqrt = S.vsel(
                                S.vneg(sqrt_abs),
                                positive_sqrt,
                                S.vcmps(normalized_dot, 0.0, full, "lt"),
                            )
                            sigmoid_exp = S.vexpdif(zero, S.vabs(signed_sqrt))
                            sigmoid_denom = S.vadds(sigmoid_exp, 1.0)
                            sigmoid_neg = S.vdiv(sigmoid_exp, sigmoid_denom)
                            sigmoid_pos = S.vsub(one, sigmoid_neg)
                            # Align the last FP32 ULP with the eager sigmoid path in
                            # the two positive transition regions that can change BF16 output.
                            positive_low = S.vsel(
                                S.vadds(sigmoid_pos, 4.0e-8),
                                sigmoid_pos,
                                S.vcmps(sigmoid_pos, 0.62, full, "lt"),
                            )
                            positive_gate = S.vsel(
                                S.vadds(sigmoid_pos, -4.0e-8),
                                positive_low,
                                S.vcmps(sigmoid_pos, 0.76, full, "gt"),
                            )
                            gate = S.vsel(
                                sigmoid_neg,
                                positive_gate,
                                S.vcmps(signed_sqrt, 0.0, full, "lt"),
                            )

                            if save_for_backward:
                                S.vsts(dot_ub[local_head], raw_dot, one_lane, dist="ONEPT_B32")
                                S.vsts(gate_ub[local_head], gate, one_lane, dist="ONEPT_B32")
                                S.vsts(rstd_x_ub[local_head], rstd_x, one_lane, dist="ONEPT_B32")
                                S.vsts(rstd_k_ub[local_head], rstd_k, one_lane, dist="ONEPT_B32")

                            for vector in range(num_vectors):
                                offset = vector * _VEC
                                x_vec = S.vcvt(
                                    S.vld(x_ub[local_head, offset], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                v_vec = S.vcvt(
                                    S.vld(v_ub[offset], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                out_vec = S.vadd(x_vec, S.vmul(gate, v_vec))
                                S.vsts(
                                    out_ub[local_head, offset],
                                    S.vcvt(out_vec, T.bfloat16),
                                    dist="PK_B32",
                                )

                    T.copy(
                        out_ub[:heads_per_task, :hidden_size],
                        output[token, head_base : head_base + heads_per_task, :],
                    )
                    if save_for_backward:
                        T.copy(dot_ub[:heads_per_task], dot_out[token, head_base : head_base + heads_per_task])
                        T.copy(gate_ub[:heads_per_task], gate_out[token, head_base : head_base + heads_per_task])
                        T.copy(rstd_x_ub[:heads_per_task], rstd_x_out[token, head_base : head_base + heads_per_task])
                        T.copy(rstd_k_ub[:heads_per_task], rstd_k_out[token, head_base : head_base + heads_per_task])

    return kernel


@tilelang.jit
def get_engram_gate_bwd_kernel_asc(
    hidden_size: int,
    scalar: float,
    k_stride_s: int,
    k_stride_h: int,
    v_stride_s: int,
    num_cores: int,
    num_grad_v_cores: int,
    clamp_value: float = 1e-6,
    hc_mult: int = 4,
):
    """Fused backward with disjoint main-gradient and grad-v vector-core roles."""
    assert hidden_size % _VEC == 0
    assert num_grad_v_cores > 0
    assert hc_mult == 4, "optimized Ascend SIMD backward requires hc_mult == 4"
    heads_per_core = 2
    assert (num_cores - num_grad_v_cores) % (hc_mult // heads_per_core) == 0

    num_tokens = T.dynamic("num_tokens")
    num_vecs = hidden_size // _VEC
    num_main_cores = num_cores - num_grad_v_cores
    num_head_groups = hc_mult // heads_per_core
    num_group_cores = num_main_cores // num_head_groups
    num_stages = 2
    # SimdVF is inlined and does not reserve the 32 KiB SIMT thread context;
    # TileLang's Ascend scheduler therefore exposes the physical 248 KiB UB.
    # Four two-stage BF16 vectors plus two two-head FP32 vectors cost 32 * D;
    # the two versions of the eight-value FP32 stats buffer add 64 bytes.
    simdvf_ub_bytes = 32 * hidden_size + 2 * 8 * 4
    assert simdvf_ub_bytes <= 248 * 1024, "Engram gate backward exceeds the 248 KiB SimdVF UB capacity"

    @T.prim_func
    def engram_gate_bwd_kernel_asc(
        grad_out: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        hidden_states: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        k: T.StridedTensor[
            (num_tokens, hc_mult, hidden_size),
            (k_stride_s, k_stride_h, 1),
            T.bfloat16,
        ],
        v: T.StridedTensor[(num_tokens, hidden_size), (v_stride_s, 1), T.bfloat16],
        weight_fused: T.Tensor[(hc_mult, hidden_size), T.float32],
        dot_in: T.Tensor[(num_tokens, hc_mult), T.float32],
        gate_in: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_x_in: T.Tensor[(num_tokens, hc_mult), T.float32],
        rstd_k_in: T.Tensor[(num_tokens, hc_mult), T.float32],
        grad_x: T.Tensor[(num_tokens, hc_mult, hidden_size), T.bfloat16],
        grad_k: T.StridedTensor[
            (num_tokens, hc_mult, hidden_size),
            (k_stride_s, k_stride_h, 1),
            T.bfloat16,
        ],
        grad_v: T.StridedTensor[(num_tokens, hidden_size), (v_stride_s, 1), T.bfloat16],
        grad_w_partial: T.Tensor[(num_group_cores, hc_mult, hidden_size), T.float32],
    ):
        with T.Kernel(num_cores) as core_id:
            go_ub = T.alloc_shared((hidden_size,), T.bfloat16)
            x_ub = T.alloc_shared((hidden_size,), T.bfloat16)
            k_ub = T.alloc_shared((hidden_size,), T.bfloat16)
            v_ub = T.alloc_shared((hidden_size,), T.bfloat16)
            weight_ub = T.alloc_shared((heads_per_core, hidden_size), T.float32)
            grad_w_ub = T.alloc_shared((heads_per_core, hidden_size), T.float32)
            stats_ub = T.alloc_shared((8,), T.float32)

            T.annotate_buffer_versions(
                {
                    go_ub: num_stages,
                    x_ub: num_stages,
                    k_ub: num_stages,
                    v_ub: num_stages,
                    stats_ub: num_stages,
                }
            )

            if core_id < num_main_cores:
                head_group_id = core_id % num_head_groups
                group_core_id = core_id // num_head_groups
                head_base = head_group_id * heads_per_core

                T.copy(
                    weight_fused[head_base : head_base + heads_per_core, :],
                    weight_ub,
                )
                with T.SimdVF():
                    zero = S.vdup(0.0, T.float32)
                    for local_head_id, vec_id in T.Parallel(heads_per_core, num_vecs):
                        S.vsts(grad_w_ub[local_head_id, vec_id * _VEC], zero)

                for token_id in T.Persistent(
                    [num_tokens],
                    num_group_cores,
                    group_core_id,
                    group_size=1,
                    num_stages=num_stages,
                ):
                    T.copy(v[token_id, :], v_ub)
                    for local_head_id in T.serial(heads_per_core):
                        head_id = head_base + local_head_id
                        T.copy(grad_out[token_id, head_id, :], go_ub)
                        T.copy(hidden_states[token_id, head_id, :], x_ub)
                        T.copy(k[token_id, head_id, :], k_ub)
                        stats_ub[0] = dot_in[token_id, head_id]
                        stats_ub[1] = gate_in[token_id, head_id]
                        stats_ub[2] = rstd_x_in[token_id, head_id]
                        stats_ub[3] = rstd_k_in[token_id, head_id]

                        with T.SimdVF():
                            reduce_zero = S.vdup(0.0, T.float32)
                            dldg = S.alloc_var(T.float32)
                            dldg = reduce_zero

                            for vec_id in range(num_vecs):
                                col = vec_id * _VEC
                                go_reg = S.vcvt(
                                    S.vld(go_ub[col], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                v_reg = S.vcvt(
                                    S.vld(v_ub[col], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                S.vmula(dldg, go_reg, v_reg)

                            S.vsts(
                                stats_ub[4],
                                S.vcadd(dldg),
                                dist="ONEPT_B32",
                            )

                        with T.SimdVF():
                            grad_w_acc = S.alloc_var(T.float32)
                            grad_x_acc = S.alloc_var(T.float32)
                            grad_k_acc = S.alloc_var(T.float32)
                            compute_zero = S.vdup(0.0, T.float32)
                            compute_one = S.vdup(1.0, T.float32)
                            dldg_broadcast = S.vld(stats_ub[4], dist="BRC_B32")
                            raw_dot = S.vld(stats_ub[0], dist="BRC_B32")
                            gate = S.vld(stats_ub[1], dist="BRC_B32")
                            rstd_x_reg = S.vld(stats_ub[2], dist="BRC_B32")
                            rstd_k_reg = S.vld(stats_ub[3], dist="BRC_B32")
                            abs_raw_dot = S.vabs(raw_dot)
                            normalized_abs = S.vmuls(
                                S.vmul(
                                    S.vmul(abs_raw_dot, rstd_x_reg),
                                    rstd_k_reg,
                                ),
                                scalar,
                            )
                            derivative_base = S.vmuls(
                                S.vmul(
                                    S.vmul(dldg_broadcast, gate),
                                    S.vsub(compute_one, gate),
                                ),
                                0.5,
                            )
                            derivative_unclamped = S.vmul(
                                derivative_base,
                                S.vsqrt(
                                    S.vdiv(
                                        S.vmuls(
                                            S.vmul(rstd_x_reg, rstd_k_reg),
                                            scalar,
                                        ),
                                        S.vmaxs(abs_raw_dot, 1e-30),
                                    )
                                ),
                            )
                            clamped_mask = S.vcmps(normalized_abs, clamp_value, op="lt")
                            derivative = S.vsel(
                                compute_zero,
                                derivative_unclamped,
                                clamped_mask,
                            )
                            dot_x = S.vmuls(
                                S.vmul(
                                    raw_dot,
                                    S.vmul(rstd_x_reg, rstd_x_reg),
                                ),
                                1.0 / hidden_size,
                            )
                            dot_k = S.vmuls(
                                S.vmul(
                                    raw_dot,
                                    S.vmul(rstd_k_reg, rstd_k_reg),
                                ),
                                1.0 / hidden_size,
                            )

                            negative_derivative_dot_x = S.vmuls(S.vmul(derivative, dot_x), -1.0)
                            negative_derivative_dot_k = S.vmuls(S.vmul(derivative, dot_k), -1.0)

                            for vec_id in range(num_vecs):
                                col = vec_id * _VEC
                                go_reg = S.vcvt(
                                    S.vld(go_ub[col], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                x_reg = S.vcvt(
                                    S.vld(x_ub[col], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                k_reg = S.vcvt(
                                    S.vld(k_ub[col], dist="UNPK_B16"),
                                    T.float32,
                                    part=0,
                                )
                                weight_reg = S.vld(weight_ub[local_head_id, col])
                                grad_w_acc = S.vld(grad_w_ub[local_head_id, col])
                                derivative_weight = S.vmul(derivative, weight_reg)
                                grad_x_acc = go_reg
                                S.vmula(grad_x_acc, k_reg, derivative_weight)
                                S.vmula(
                                    grad_x_acc,
                                    x_reg,
                                    negative_derivative_dot_x,
                                )
                                grad_k_acc = S.vmul(x_reg, derivative_weight)
                                S.vmula(
                                    grad_k_acc,
                                    k_reg,
                                    negative_derivative_dot_k,
                                )
                                S.vmula(
                                    grad_w_acc,
                                    S.vmul(derivative, x_reg),
                                    k_reg,
                                )
                                S.vsts(
                                    x_ub[col],
                                    S.vcvt(grad_x_acc, T.bfloat16),
                                    dist="PK_B32",
                                )
                                S.vsts(
                                    k_ub[col],
                                    S.vcvt(grad_k_acc, T.bfloat16),
                                    dist="PK_B32",
                                )
                                S.vsts(grad_w_ub[local_head_id, col], grad_w_acc)

                        T.copy(x_ub, grad_x[token_id, head_id, :])
                        T.copy(k_ub, grad_k[token_id, head_id, :])

                for local_head_id in T.serial(heads_per_core):
                    T.copy(
                        grad_w_ub[local_head_id, :],
                        grad_w_partial[group_core_id, head_base + local_head_id, :],
                    )
            else:
                grad_v_core_id = core_id - num_main_cores
                for token_id in T.Persistent(
                    [num_tokens],
                    num_grad_v_cores,
                    grad_v_core_id,
                    group_size=1,
                    num_stages=num_stages,
                ):
                    T.copy(grad_out[token_id, 0, :], go_ub)
                    T.copy(grad_out[token_id, 1, :], x_ub)
                    T.copy(grad_out[token_id, 2, :], k_ub)
                    T.copy(grad_out[token_id, 3, :], v_ub)
                    T.copy(gate_in[token_id, :], stats_ub[:hc_mult])

                    with T.SimdVF():
                        gate_0 = S.vld(stats_ub[0], dist="BRC_B32")
                        gate_1 = S.vld(stats_ub[1], dist="BRC_B32")
                        gate_2 = S.vld(stats_ub[2], dist="BRC_B32")
                        gate_3 = S.vld(stats_ub[3], dist="BRC_B32")
                        for vec_id in range(num_vecs):
                            col = vec_id * _VEC
                            go_0 = S.vcvt(
                                S.vld(go_ub[col], dist="UNPK_B16"),
                                T.float32,
                                part=0,
                            )
                            go_1 = S.vcvt(
                                S.vld(x_ub[col], dist="UNPK_B16"),
                                T.float32,
                                part=0,
                            )
                            go_2 = S.vcvt(
                                S.vld(k_ub[col], dist="UNPK_B16"),
                                T.float32,
                                part=0,
                            )
                            go_3 = S.vcvt(
                                S.vld(v_ub[col], dist="UNPK_B16"),
                                T.float32,
                                part=0,
                            )
                            grad_v_reg = S.vadd(
                                S.vadd(
                                    S.vmul(go_0, gate_0),
                                    S.vmul(go_1, gate_1),
                                ),
                                S.vadd(
                                    S.vmul(go_2, gate_2),
                                    S.vmul(go_3, gate_3),
                                ),
                            )
                            S.vsts(
                                go_ub[col],
                                S.vcvt(grad_v_reg, T.bfloat16),
                                dist="PK_B32",
                            )

                    T.copy(go_ub, grad_v[token_id, :])

    return engram_gate_bwd_kernel_asc
