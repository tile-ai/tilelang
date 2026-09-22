import tilelang
from tilelang import language as T
from tilelang.language import simd as S

from tile_kernels.config import get_num_vec_cores
from tile_kernels.quant.common import CastOutputConfig, get_sf_shape


@tilelang.jit
def get_swiglu_forward_kernel_asc(
    hidden: int,
    num_experts: int,
    alignment: int,
    with_weight: bool,
    with_routed_scaling: bool,
    clamp_value,
    count_clamp: bool,
    in_dtype: T.dtype,
    out_config: CastOutputConfig,
    num_cores: int,
):
    out_dtype = out_config.dtype
    is_bf16_in = in_dtype == T.bfloat16
    is_bf16_out = out_dtype == T.bfloat16
    has_clamp = clamp_value is not None
    has_psum = num_experts > 0

    num_stages = 3
    vec_size = 64
    assert hidden % vec_size == 0, f'Ascend swiglu requires hidden % {vec_size} == 0, got {hidden}'
    num_vecs = hidden // vec_size

    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    sf_stride = T.dynamic('sf_stride')
    sf_shape = (1, 1)  # placeholder, no scaling factors

    @T.prim_func
    def swiglu_forward_kernel(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), in_dtype],
        out: T.Tensor[(num_expanded_tokens, hidden), out_dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
        psum_num_tokens_per_expert: T.Tensor[num_experts, T.int32],
        topk_weights: T.Tensor[(num_expanded_tokens,), T.float32],
        routed_scaling_factor: T.float32,
        clamped_count: T.Tensor[(4,), T.int64],
    ):
        with T.Kernel(num_cores) as core_id:
            if count_clamp:
                acc_ub = T.alloc_shared((4,), T.int32)
                pacc_ub = T.alloc_shared((3, vec_size), T.float32)  # per lane accumulator pacc_ub. use vcadd at the last moment

                for i in T.serial(4):
                    acc_ub[i] = 0
                with T.SimdVF():
                    T.clear(pacc_ub)

            if has_psum:
                psum_ub = T.alloc_shared((num_experts,), T.int32)
                T.copy(psum_num_tokens_per_expert, psum_ub)
                expert_id = T.alloc_var(T.int32, init=0)

            xl_ub = T.alloc_shared((hidden,), in_dtype)
            xr_ub = T.alloc_shared((hidden,), in_dtype)
            out_ub = T.alloc_shared((hidden,), out_dtype)
            T.annotate_buffer_versions({xl_ub: num_stages, xr_ub: num_stages, out_ub: num_stages})

            for pid in T.Persistent([num_expanded_tokens], num_cores, core_id, group_size=1, num_stages=num_stages):
                token_id = T.alloc_var(T.int32, init=pid)
                valid = T.alloc_var(T.bool, init=True)
                if has_psum:
                    # Tokens are walked in increasing order per core, so advance a
                    # monotonic expert cursor instead of a per-token binary search.
                    while expert_id < num_experts and token_id >= (psum_ub[expert_id] + alignment - 1) // alignment * alignment:
                        expert_id += 1
                    valid = (expert_id < num_experts) and (token_id < psum_ub[expert_id])

                if valid:
                    weight = T.alloc_var(T.float32, init=1.0)
                    if with_weight:
                        weight = topk_weights[token_id]
                        if with_routed_scaling:
                            weight = weight * routed_scaling_factor

                    T.copy(x[token_id, 0], xl_ub)
                    T.copy(x[token_id, hidden], xr_ub)

                    with T.SimdVF():
                        ones = S.vdup(1.0, T.float32)
                        zeros = S.vdup(0.0, T.float32)

                        if has_clamp and count_clamp:
                            acc = S.alloc_local((3,), T.float32)
                            acc[0] = zeros
                            acc[1] = zeros
                            acc[2] = zeros

                        for c in T.serial(num_vecs):
                            col = c * vec_size
                            if is_bf16_in:
                                val_l = S.vcvt(S.vld(xl_ub[col], dist='UNPK_B16'), T.float32, part=0)
                                val_r = S.vcvt(S.vld(xr_ub[col], dist='UNPK_B16'), T.float32, part=0)
                            else:
                                val_l = S.vld(xl_ub[col])
                                val_r = S.vld(xr_ub[col])
                            if has_clamp:
                                if count_clamp:
                                    # Count clamped elements with a single merging add per
                                    # predicate (acc += 1 on lanes where the compare is true).
                                    clamped_mask_0 = S.vcmps(val_l, clamp_value, op='gt')
                                    clamped_mask_1 = S.vcmps(val_r, clamp_value, op='gt')
                                    clamped_mask_2 = S.vcmps(val_r, -clamp_value, op='lt')
                                    acc[0] = S.vadds(acc[0], 1.0, clamped_mask_0, mode='MODE_MERGING')
                                    acc[1] = S.vadds(acc[1], 1.0, clamped_mask_1, mode='MODE_MERGING')
                                    acc[2] = S.vadds(acc[2], 1.0, clamped_mask_2, mode='MODE_MERGING')

                                val_l = S.vmins(val_l, clamp_value)
                                val_r = S.vmaxs(S.vmins(val_r, clamp_value), -clamp_value)
                            # silu(l) * r * weight = l / (1 + exp(-l)) * r * weight
                            sig = S.vdiv(val_l, S.vadd(S.vexpdif(zeros, val_l), ones))
                            val = S.vmul(sig, val_r)
                            if with_weight:
                                val = S.vmuls(val, weight)
                            if is_bf16_out:
                                S.vsts(out_ub[col], S.vcvt(val, T.bfloat16), dist='PK_B32')
                            else:
                                S.vsts(out_ub[col], val)

                        if has_clamp and count_clamp:
                            for i in T.unroll(3, explicit=True):
                                S.vsts(pacc_ub[i, 0], S.vadd(S.vld(pacc_ub[i, 0]), acc[i]))

                    T.copy(out_ub, out[token_id, 0])

                    if count_clamp:
                        acc_ub[3] += hidden

            if count_clamp:
                with T.SimdVF():
                    for i in T.unroll(3, explicit=True):
                        S.vsts(acc_ub[i], S.vcvt(S.vcadd(S.vld(pacc_ub[i, 0])), T.int32), dist='ONEPT_B32')

                with T.SimtVF(threads=1):
                    # TODO: check if int32 is feasible for faster atomic add
                    for i in T.serial(4):
                        T.atomic_add(clamped_count[i], T.int64(acc_ub[i]))

    return swiglu_forward_kernel


@tilelang.jit
def get_swiglu_forward_and_per_token_cast_kernel_asc(
    hidden: int,
    num_experts: int,
    alignment: int,
    with_weight: bool,
    with_routed_scaling: bool,
    clamp_value,
    count_clamp: bool,
    in_dtype: T.dtype,
    out_config: CastOutputConfig,
    num_cores: int,
):
    """Fuse SwiGLU with the raw grouped-output pattern from per_token_cast."""
    group_size = out_config.sf_block[1]
    assert out_config.dtype in (T.float8_e4m3fn, T.float4_e2m1fn)
    assert out_config.sf_block[0] == 1
    assert group_size in (32, 128)
    assert hidden % 64 == 0

    is_bf16_in = in_dtype == T.bfloat16
    is_fp4 = out_config.dtype == T.float4_e2m1fn
    has_clamp = clamp_value is not None
    has_psum = num_experts > 0
    is_packed_sf = out_config.use_packed_ue8m0
    is_col_major_sf = out_config.use_tma_aligned_col_major_sf
    num_groups = (hidden + group_size - 1) // group_size
    num_sf_batches = (num_groups + 63) // 64
    num_sf_slots = num_sf_batches * 64
    sf_ub_dtype = T.uint8 if is_packed_sf else T.float32
    num_stages = 2
    vec_size = 64
    num_vecs = hidden // vec_size
    quant_max = 6.0 if is_fp4 else 448.0

    num_expanded_tokens = T.dynamic('num_expanded_tokens')
    sf_stride = T.dynamic('sf_stride')
    sf_shape = get_sf_shape((num_expanded_tokens, hidden), out_config)

    @T.prim_func
    def swiglu_forward_and_per_token_cast_kernel(
        x: T.Tensor[(num_expanded_tokens, hidden * 2), in_dtype],
        out: T.Tensor[(num_expanded_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[sf_shape, (sf_stride, 1), out_config.sf_dtype],
        psum_num_tokens_per_expert: T.Tensor[num_experts, T.int32],
        topk_weights: T.Tensor[(num_expanded_tokens,), T.float32],
        routed_scaling_factor: T.float32,
        clamped_count: T.Tensor[(4,), T.int64],
    ):
        with T.Kernel(num_cores) as core_id:
            if count_clamp:
                acc_ub = T.alloc_shared((4,), T.int32)
                pacc_ub = T.alloc_shared((3, vec_size), T.float32)
                for i in T.serial(4):
                    acc_ub[i] = 0
                with T.SimdVF():
                    T.clear(pacc_ub)

            if has_psum:
                psum_ub = T.alloc_shared((num_experts,), T.int32)
                T.copy(psum_num_tokens_per_expert, psum_ub)
                expert_id = T.alloc_var(T.int32, init=0)

            xl_ub = T.alloc_shared((hidden,), in_dtype)
            xr_ub = T.alloc_shared((hidden,), in_dtype)
            value_ub = T.alloc_shared((hidden,), T.float32)
            out_ub = T.alloc_shared((((hidden + 127) // 128) * 128 if is_fp4 else hidden,), out_config.dtype)
            amax_ub = T.alloc_shared((num_sf_slots,), T.float32)
            sf_storage_ub = T.alloc_shared((num_sf_slots,), sf_ub_dtype)
            sf_inv_ub = T.alloc_shared((num_sf_slots,), T.float32)
            sf_linear_ub = T.Tensor((num_sf_slots,), sf_ub_dtype, sf_storage_ub.data)
            T.annotate_buffer_versions(
                {
                    xl_ub: num_stages,
                    xr_ub: num_stages,
                    value_ub: num_stages,
                    out_ub: num_stages,
                    amax_ub: num_stages,
                    sf_storage_ub: num_stages,
                    sf_inv_ub: num_stages,
                }
            )

            for pid in T.Persistent(
                [num_expanded_tokens],
                num_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                token_id = T.alloc_var(T.int32, init=pid)
                valid = T.alloc_var(T.bool, init=True)
                if has_psum:
                    while expert_id < num_experts and token_id >= (psum_ub[expert_id] + alignment - 1) // alignment * alignment:
                        expert_id += 1
                    valid = (expert_id < num_experts) and (token_id < psum_ub[expert_id])

                if valid:
                    weight = T.alloc_var(T.float32, init=1.0)
                    if with_weight:
                        weight = topk_weights[token_id]
                        if with_routed_scaling:
                            weight = weight * routed_scaling_factor

                    T.copy(x[token_id, 0], xl_ub)
                    T.copy(x[token_id, hidden], xr_ub)

                    with T.SimdVF():
                        ones = S.vdup(1.0, T.float32)
                        zeros = S.vdup(0.0, T.float32)
                        if has_clamp and count_clamp:
                            acc = S.alloc_local((3,), T.float32)
                            acc[0] = zeros
                            acc[1] = zeros
                            acc[2] = zeros

                        for vector in T.serial(num_vecs):
                            col = vector * vec_size
                            if is_bf16_in:
                                val_l = S.vcvt(S.vld(xl_ub[col], dist='UNPK_B16'), T.float32, part=0)
                                val_r = S.vcvt(S.vld(xr_ub[col], dist='UNPK_B16'), T.float32, part=0)
                            else:
                                val_l = S.vld(xl_ub[col])
                                val_r = S.vld(xr_ub[col])
                            if has_clamp:
                                if count_clamp:
                                    clamped_mask_0 = S.vcmps(val_l, clamp_value, op='gt')
                                    clamped_mask_1 = S.vcmps(val_r, clamp_value, op='gt')
                                    clamped_mask_2 = S.vcmps(val_r, -clamp_value, op='lt')
                                    acc[0] = S.vadds(acc[0], 1.0, clamped_mask_0, mode='MODE_MERGING')
                                    acc[1] = S.vadds(acc[1], 1.0, clamped_mask_1, mode='MODE_MERGING')
                                    acc[2] = S.vadds(acc[2], 1.0, clamped_mask_2, mode='MODE_MERGING')
                                val_l = S.vmins(val_l, clamp_value)
                                val_r = S.vmaxs(S.vmins(val_r, clamp_value), -clamp_value)
                            val = S.vmul(S.vdiv(val_l, S.vadd(S.vexpdif(zeros, val_l), ones)), val_r)
                            if with_weight:
                                S.vsts(value_ub[col], S.vmuls(val, weight))
                            else:
                                S.vsts(value_ub[col], val)

                        if has_clamp and count_clamp:
                            for i in T.unroll(3, explicit=True):
                                S.vsts(pacc_ub[i, 0], S.vadd(S.vld(pacc_ub[i, 0]), acc[i]))

                    with T.SimdVF():
                        for sf_batch in T.unroll(num_sf_batches, explicit=True):
                            S.vsts(amax_ub[sf_batch * 64], S.vdup(0.0, T.float32))

                    with T.SimdVF():
                        if group_size == 32:
                            mask_low = S.pset(32, 'PAT_VL32')
                            mask_high = S.vcmps(S.vci(0, T.int32), 31, op='gt')
                            for vector in T.serial(num_vecs):
                                values = S.vabs(S.vld(value_ub[vector * vec_size]))
                                S.vsts(amax_ub[vector * 2], S.vcmax(values, mask_low), dist='ONEPT_B32')
                                S.vsts(amax_ub[vector * 2 + 1], S.vcmax(values, mask_high), dist='ONEPT_B32')
                        else:
                            if hidden >= group_size:
                                for group in T.serial(hidden // group_size):
                                    col = group * group_size
                                    values0 = S.vabs(S.vld(value_ub[col]))
                                    values1 = S.vabs(S.vld(value_ub[col + vec_size]))
                                    S.vsts(amax_ub[group], S.vcmax(S.vmax(values0, values1)), dist='ONEPT_B32')
                            if hidden % group_size:
                                tail_col = (num_groups - 1) * group_size
                                values = S.vabs(S.vld(value_ub[tail_col]))
                                S.vsts(amax_ub[num_groups - 1], S.vcmax(values), dist='ONEPT_B32')

                    with T.SimdVF():
                        for sf_batch in T.serial(num_sf_batches):
                            sf_offset = sf_batch * 64
                            amax = S.vmaxs(S.vld(amax_ub[sf_offset]), out_config.clamp_min_value)
                            if out_config.round_sf:
                                scale_raw = S.vmuls(amax, 1.0 / quant_max)
                                scale_bits = T.reinterpret(scale_raw, 'uint32x64')
                                scale_exp = S.vadds(S.vshrs(S.vsub(scale_bits, S.vdup(1, T.uint32)), 23), 1)
                                inv_exp = S.vsub(S.vdup(254, T.uint32), scale_exp)
                                scale_inv = T.reinterpret(S.vshls(inv_exp, 23), 'float32x64')
                                if is_packed_sf:
                                    S.vsts(sf_linear_ub[sf_offset], scale_exp, dist='PK4_B32')
                                else:
                                    scale = T.reinterpret(S.vshls(scale_exp, 23), 'float32x64')
                                    S.vsts(sf_linear_ub[sf_offset], scale)
                            else:
                                quant_max_vec = S.vdup(quant_max, T.float32)
                                scale = S.vdiv(amax, quant_max_vec)
                                scale_inv = S.vdiv(quant_max_vec, amax)
                                S.vsts(sf_linear_ub[sf_offset], scale)
                            S.vsts(sf_inv_ub[sf_offset], scale_inv)

                    with T.SimdVF():
                        if is_fp4:
                            scaled = S.alloc_local((2,), T.float32)
                            if group_size == 32:
                                mask_low = S.pset(32, 'PAT_VL32')
                            for pair in T.serial(num_vecs // 2):
                                for half in T.unroll(2, explicit=True):
                                    vector = pair * 2 + half
                                    if group_size == 32:
                                        inverse_low = S.vld(sf_inv_ub[vector * 2], dist='BRC_B32')
                                        inverse_high = S.vld(sf_inv_ub[vector * 2 + 1], dist='BRC_B32')
                                        inverse = S.vsel(inverse_low, inverse_high, mask_low)
                                    else:
                                        inverse = S.vld(sf_inv_ub[pair], dist='BRC_B32')
                                    scaled[half] = S.vmul(S.vld(value_ub[vector * vec_size]), inverse)
                                low, high = S.vdintlv(T.reinterpret(scaled[0], 'uint16x128'), T.reinterpret(scaled[1], 'uint16x128'))
                                scaled_bf16 = T.reinterpret(S.vor(high, S.vmins(low, 1)), 'bfloat16x128')
                                S.vsts(out_ub[pair * 128], S.vcvt(scaled_bf16, T.float4_e2m1fn), dist='PK4_B32')
                            if num_vecs % 2:
                                tail_vector = num_vecs - 1
                                if group_size == 32:
                                    tail_inverse_low = S.vld(sf_inv_ub[tail_vector * 2], dist='BRC_B32')
                                    tail_inverse_high = S.vld(sf_inv_ub[tail_vector * 2 + 1], dist='BRC_B32')
                                    tail_inverse = S.vsel(tail_inverse_low, tail_inverse_high, mask_low)
                                else:
                                    tail_inverse = S.vld(sf_inv_ub[tail_vector // 2], dist='BRC_B32')
                                tail = S.vmul(S.vld(value_ub[tail_vector * vec_size]), tail_inverse)
                                tail_low, tail_high = S.vdintlv(
                                    T.reinterpret(tail, 'uint16x128'), T.reinterpret(S.vdup(0.0, T.float32), 'uint16x128')
                                )
                                tail_bf16 = T.reinterpret(S.vor(tail_high, S.vmins(tail_low, 1)), 'bfloat16x128')
                                S.vsts(out_ub[tail_vector * vec_size], S.vcvt(tail_bf16, T.float4_e2m1fn), dist='PK4_B32')
                        elif group_size == 32:
                            mask_low = S.pset(32, 'PAT_VL32')
                            for vector in T.serial(num_vecs):
                                inverse_low = S.vld(sf_inv_ub[vector * 2], dist='BRC_B32')
                                inverse_high = S.vld(sf_inv_ub[vector * 2 + 1], dist='BRC_B32')
                                inverse = S.vsel(inverse_low, inverse_high, mask_low)
                                scaled = S.vmul(S.vld(value_ub[vector * vec_size]), inverse)
                                S.vsts(out_ub[vector * vec_size], S.vcvt(scaled, T.float8_e4m3fn), dist='PK4_B32')
                        else:
                            for vector in T.serial(num_vecs):
                                inverse = S.vld(sf_inv_ub[vector // 2], dist='BRC_B32')
                                scaled = S.vmul(S.vld(value_ub[vector * vec_size]), inverse)
                                S.vsts(out_ub[vector * vec_size], S.vcvt(scaled, T.float8_e4m3fn), dist='PK4_B32')

                    # A single token produces a short SF row. Scalar stores avoid
                    # issuing an unaligned, sub-32-byte strided MTE transfer.
                    with T.SimtVF(threads=num_groups):
                        group = T.get_thread_binding()
                        if is_col_major_sf:
                            if is_packed_sf:
                                out_sf[group // 2, token_id * 2 + group % 2] = sf_linear_ub[group]
                            else:
                                out_sf[group, token_id] = sf_linear_ub[group]
                        else:
                            out_sf[token_id, group] = sf_linear_ub[group]
                    T.copy(out_ub[:hidden], out[token_id, 0:hidden])

                    if count_clamp:
                        acc_ub[3] += hidden

            if count_clamp:
                with T.SimdVF():
                    for i in T.unroll(3, explicit=True):
                        S.vsts(acc_ub[i], S.vcvt(S.vcadd(S.vld(pacc_ub[i, 0])), T.int32), dist='ONEPT_B32')
                with T.SimtVF(threads=1):
                    for i in T.serial(4):
                        T.atomic_add(clamped_count[i], T.int64(acc_ub[i]))

    return swiglu_forward_and_per_token_cast_kernel
