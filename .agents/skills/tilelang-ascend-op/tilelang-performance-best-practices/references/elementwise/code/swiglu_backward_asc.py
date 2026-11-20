import tilelang
from tilelang import language as T
from tilelang.language import simd as S

from tile_kernels.config import get_num_vec_cores
from tile_kernels.utils import align


@tilelang.jit
def get_swiglu_backward_kernel_asc(
    hidden: int,
    num_experts: int,
    alignment: int,
    with_weight: bool,
    with_routed_scaling: bool,
    use_clamp: bool,
    x_dtype: T.dtype,
    act_x_grad_dtype: T.dtype,
    out_dtype: T.dtype,
    do_recompute: bool = False,
):
    num_cores = get_num_vec_cores()
    num_stages = 2
    vec_size = 64  # SIMD vector length (float32 lanes per vreg)

    # Stage the whole token_id at once: out-of-range tail lanes are masked out of the
    # weight_grad reduction so the per-token_id sum stays exact.
    # Align to 2 * vec_size so num_vregs is even for 2-way unrolling.
    hidden_aligned = align(hidden, 2 * vec_size)
    num_vregs = hidden_aligned // vec_size

    is_bf16_in = x_dtype == T.bfloat16
    is_bf16_grad_in = act_x_grad_dtype == T.bfloat16
    is_bf16_out = out_dtype == T.bfloat16

    num_expanded_tokens = T.dynamic("num_expanded_tokens")

    @T.prim_func
    def swiglu_backward_kernel_asc(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), x_dtype],
        act_x_grad: T.Tensor[(num_expanded_tokens, hidden), act_x_grad_dtype],
        topk_weights: T.Tensor[(num_expanded_tokens,), T.float32],
        routed_scaling_factor: T.float32,
        psum_num_tokens_per_expert: T.Tensor[(num_experts,), T.int32],
        x_grad: T.Tensor[(num_expanded_tokens, hidden * 2), out_dtype],
        weight_grad: T.Tensor[(num_expanded_tokens,), T.float32],
        out: T.Tensor[(num_expanded_tokens, hidden), out_dtype],
        clamp_value: T.float32,
    ):
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((hidden_aligned,), x_dtype)
            y_ub = T.alloc_shared((hidden_aligned,), x_dtype)
            act_x_grad_ub = T.alloc_shared((hidden_aligned,), act_x_grad_dtype)
            x_grad_ub = T.alloc_shared((hidden_aligned,), out_dtype)
            y_grad_ub = T.alloc_shared((hidden_aligned,), out_dtype)
            w_grad_ub = T.alloc_shared((vec_size,), T.float32)
            if do_recompute:
                act_out_ub = T.alloc_shared((hidden_aligned,), out_dtype)
            if num_experts:
                psum_ub = T.alloc_shared((num_experts,), T.int32)

            buffer_versions = {
                x_ub: num_stages,
                y_ub: num_stages,
                act_x_grad_ub: num_stages,
                x_grad_ub: num_stages,
                y_grad_ub: num_stages,
            }
            if with_weight:
                buffer_versions[w_grad_ub] = num_stages
            if do_recompute:
                buffer_versions[act_out_ub] = num_stages
            T.annotate_buffer_versions(buffer_versions)

            if num_experts:
                T.copy(psum_num_tokens_per_expert, psum_ub[:num_experts])

            for token_id in T.Persistent([num_expanded_tokens], num_cores, core_id, group_size=1, num_stages=num_stages):
                if token_id < num_expanded_tokens:
                    w_var = T.alloc_var(T.float32, init=1.0)
                    if with_weight:
                        keep = T.alloc_var(T.float32, init=1.0)
                        if num_experts:
                            for e in T.unroll(num_experts):
                                end = psum_ub[e]
                                keep = keep * T.cast(not (end <= token_id and token_id < align(end, alignment)), T.float32)
                        w_var = topk_weights[token_id] * keep
                        if with_routed_scaling:
                            w_var = w_var * routed_scaling_factor

                    T.copy(x[token_id, 0:hidden], x_ub[:hidden])
                    T.copy(x[token_id, hidden : 2 * hidden], y_ub[:hidden])
                    T.copy(act_x_grad[token_id, 0:hidden], act_x_grad_ub[:hidden])

                    with T.SimdVF():
                        one_mask = S.pset(32, "PAT_VL1")

                        ones = S.vdup(1.0, T.float32)
                        zeros = S.vdup(0.0, T.float32)
                        w_reg = S.vdup(w_var, T.float32)

                        # Loop-carried partial sum for the per-token_id weight_grad
                        if with_weight:
                            hidden_reg = S.vdup(T.int32(hidden), T.int32)
                            w_grad_acc = S.alloc_var(T.float32)
                            w_grad_acc = zeros

                        # Pre-allocate vector register arrays for 2-way parallel processing
                        x_local = S.alloc_local((2,), T.float32)
                        y_local = S.alloc_local((2,), T.float32)
                        act_x_grad_local = S.alloc_local((2,), T.float32)
                        act_out = S.alloc_local((2,), T.float32)
                        g_ws_local = S.alloc_local((2,), T.float32)
                        x_grad_local = S.alloc_local((2,), T.float32)
                        y_grad_local = S.alloc_local((2,), T.float32)
                        if use_clamp:
                            is_clamped_x = T.alloc_local((2,), "boolx256")
                            is_clamped_y = T.alloc_local((2,), "boolx256")

                        for v in T.serial(num_vregs // 2):
                            for i in T.unroll(2, explicit=True):
                                col = (v * 2 + i) * vec_size

                                # load from global memory
                                if is_bf16_in:
                                    x_local[i] = S.vcvt(S.vld(x_ub[col], dist="UNPK_B16"), T.float32, part=0)
                                    y_local[i] = S.vcvt(S.vld(y_ub[col], dist="UNPK_B16"), T.float32, part=0)
                                else:
                                    x_local[i] = S.vld(x_ub[col])
                                    y_local[i] = S.vld(y_ub[col])
                                if is_bf16_grad_in:
                                    act_x_grad_local[i] = S.vcvt(S.vld(act_x_grad_ub[col], dist="UNPK_B16"), T.float32, part=0)
                                else:
                                    act_x_grad_local[i] = S.vld(act_x_grad_ub[col])

                                if use_clamp:
                                    is_clamped_x[i] = S.vcmps(x_local[i], clamp_value, op="gt")
                                    is_clamped_y[i] = S.vcmps(S.vabs(y_local[i]), clamp_value, op="gt")
                                    x_local[i] = S.vmins(x_local[i], clamp_value)
                                    y_local[i] = S.vmaxs(y_local[i], -clamp_value)
                                    y_local[i] = S.vmins(y_local[i], clamp_value)

                                # tmp = 1 + exp(-x); s = 1 / tmp; act_out = (x / tmp) * y.
                                tmp_reg = S.vadds(S.vexpdif(zeros, x_local[i]), 1.0)
                                s_reg = S.vdiv(ones, tmp_reg)
                                act_out[i] = S.vmul(S.vdiv(x_local[i], tmp_reg), y_local[i])

                                # weight_grad += g * act_out  (valid lanes only)
                                if with_weight:
                                    w_grad = S.vmul(act_x_grad_local[i], act_out[i])
                                    lane_id = S.vci(T.int32(col), T.int32)
                                    valid_mask = S.vcmp(lane_id, hidden_reg, op="lt")
                                    w_grad = S.vsel(w_grad, zeros, valid_mask)
                                    w_grad_acc = S.vadd(w_grad_acc, w_grad)

                                # g_ws_local = g * w * s
                                g_ws_local[i] = S.vmul(S.vmul(act_x_grad_local[i], w_reg), s_reg)

                                # x_grad = is_clamped_x ? 0 : g_ws_local * y * (1 + x * (1 - s))
                                inner_reg = S.vadd(ones, S.vmul(x_local[i], S.vsub(ones, s_reg)))
                                x_grad_local[i] = S.vmul(S.vmul(g_ws_local[i], y_local[i]), inner_reg)
                                # y_grad = is_clamped_y ? 0 : g_ws_local * x
                                y_grad_local[i] = S.vmul(g_ws_local[i], x_local[i])
                                if use_clamp:
                                    x_grad_local[i] = S.vsel(zeros, x_grad_local[i], is_clamped_x[i])
                                    y_grad_local[i] = S.vsel(zeros, y_grad_local[i], is_clamped_y[i])

                                # Store outputs
                                if is_bf16_out:
                                    S.vsts(x_grad_ub[col], S.vcvt(x_grad_local[i], T.bfloat16), dist="PK_B32")
                                    S.vsts(y_grad_ub[col], S.vcvt(y_grad_local[i], T.bfloat16), dist="PK_B32")
                                else:
                                    S.vsts(x_grad_ub[col], x_grad_local[i])
                                    S.vsts(y_grad_ub[col], y_grad_local[i])

                                if do_recompute:
                                    act_out[i] = S.vmul(act_out[i], w_reg)
                                    if is_bf16_out:
                                        S.vsts(act_out_ub[col], S.vcvt(act_out[i], T.bfloat16), dist="PK_B32")
                                    else:
                                        S.vsts(act_out_ub[col], act_out[i])

                        # Cross-lane reduce the per-token_id partial sum and store once
                        if with_weight:
                            wg_sum = S.vcadd(w_grad_acc)
                            if with_routed_scaling:
                                wg_sum = S.vmuls(wg_sum, routed_scaling_factor)
                            S.vsts(w_grad_ub[0], wg_sum, one_mask, dist="ONEPT_B32")

                    T.copy(x_grad_ub[:hidden], x_grad[token_id, 0:hidden])
                    T.copy(y_grad_ub[:hidden], x_grad[token_id, hidden : 2 * hidden])
                    if do_recompute:
                        T.copy(act_out_ub[:hidden], out[token_id, 0:hidden])
                    if with_weight:
                        T.copy(w_grad_ub[:1], weight_grad[token_id : token_id + 1])

    return swiglu_backward_kernel_asc
