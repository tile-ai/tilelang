import tilelang
from tilelang import language as T
from tilelang.language import simd as S

from tile_kernels.config import get_num_vec_cores
from tile_kernels.utils import align, ceil_div


def _balanced_vector_reduce(values, count, mask, op):
    """Build a balanced SIMD reduction tree for a static vreg count."""
    current = [values[i] for i in range(count)]
    while len(current) > 1:
        next_level = [op(current[i], current[i + 1], mask) for i in range(0, len(current) - 1, 2)]
        if len(current) % 2:
            next_level.append(current[-1])
        current = next_level
    return current[0]


def _balanced_score_index_reduce(scores, indices, count, mask):
    """Reduce score/index vreg pairs, preserving the lower index on ties."""
    current = [(scores[i], indices[i]) for i in range(count)]
    while len(current) > 1:
        next_level = []
        for i in range(0, len(current) - 1, 2):
            left_score, left_index = current[i]
            right_score, right_index = current[i + 1]
            keep_left = S.vcmp(left_score, right_score, mask, "ge")
            next_level.append((S.vmax(left_score, right_score, mask), S.vsel(left_index, right_index, keep_left)))
        if len(current) % 2:
            next_level.append(current[-1])
        current = next_level
    return current[0]


@tilelang.jit
def get_moe_topk_gate_kernel_asc(
    scoring_type: int,
    num_topk: int,
    num_routed_experts: int,
    mask_exists: bool,
    fix_routing_mask_exists: bool,
    unmapped_topk_idx_exists: bool,
    to_physical_map_exists: bool,
    has_bias: bool,
    has_image_token_mask: bool,
    has_force_random: bool,
):
    num_cores = get_num_vec_cores()
    vector_length = 64
    num_stages = 2
    # Seven FP32 vectors cover the largest tested expert axis (448) in one
    # register-resident TopK scan. Smaller expert counts still allocate only
    # ceil_div(num_routed_experts, 64) registers.
    max_vregs_per_group = 7
    max_physical_topk = num_topk + 2
    physical_map_threads = 256

    # Keep the register footprint bounded, and scan additional expert groups
    # instead of silently dropping experts beyond the first six vectors.
    num_vregs_per_group = ceil_div(min(num_routed_experts, max_vregs_per_group * vector_length), vector_length)
    experts_per_group = num_vregs_per_group * vector_length
    num_groups = ceil_div(num_routed_experts, experts_per_group)
    num_aligned_experts = num_groups * experts_per_group
    num_aligned_topk = align(max_physical_topk, vector_length)

    num_tokens = T.dynamic("num_tokens")
    num_physical_topk = T.dynamic("num_physical_topk")
    num_logical_experts = T.dynamic("num_logical_experts")
    num_duplicate_experts = T.dynamic("num_duplicate_experts")
    unmapped_topk_idx_stride = T.dynamic("unmapped_topk_idx_stride")

    @T.prim_func
    def moe_topk_gate_kernel_asc(
        logits: T.Tensor[(num_tokens, num_routed_experts), T.float32],
        bias: T.Tensor[(num_routed_experts,), T.float32],
        image_bias: T.Tensor[(num_routed_experts,), T.float32],
        image_token_mask: T.Tensor[(num_tokens,), T.bool],
        mask: T.Tensor[(num_tokens,), T.bool],
        fix_routing_mask: T.Tensor[(num_tokens,), T.bool],
        to_physical_map: T.Tensor[(num_logical_experts, num_duplicate_experts), T.int32],
        logical_count: T.Tensor[(num_logical_experts,), T.int32],
        topk_idx: T.Tensor[(num_tokens, num_physical_topk), T.int64],
        unmapped_topk_idx: T.StridedTensor[(num_tokens, num_topk), (unmapped_topk_idx_stride, 1), T.int64],
        topk_weights: T.Tensor[(num_tokens, num_physical_topk), T.float32],
        force_random: T.Tensor[(num_tokens,), T.bool],
        num_shared_experts: T.int32,
        routed_scaling_factor: T.float32,
        ep_rank: T.int32,
    ):
        with T.Kernel(num_cores) as core_id:
            logits_ub = T.alloc_shared((num_aligned_experts,), T.float32)
            bias_ub = T.alloc_shared((num_aligned_experts,), T.float32)
            scores_ub = T.alloc_shared((num_aligned_experts,), T.float32)
            ranked_scores_ub = T.alloc_shared((num_aligned_experts,), T.float32)

            # int64 output is written through an interleaved int32 view. The
            # even words are the low halves; clearing the buffer sets all high
            # halves to zero for non-negative expert indices.
            out_idx_ub = T.alloc_shared((num_aligned_topk * 2,), T.int32)
            out_weights_ub = T.alloc_shared((num_aligned_topk,), T.float32)
            random_raw_ub = T.alloc_shared((num_aligned_topk,), T.int32)
            if has_force_random and unmapped_topk_idx_exists and num_topk in (4, 6, 8):
                unmapped_fill_i32_ub = T.alloc_shared((vector_length,), T.int32)
                unmapped_fill_i64_ub = T.view(unmapped_fill_i32_ub, (vector_length // 2,), T.int64)
            out_idx_i64_ub = T.view(out_idx_ub, (num_aligned_topk,), T.int64)

            T.annotate_buffer_versions(
                {
                    logits_ub: num_stages,
                    out_idx_ub: num_stages,
                    out_weights_ub: num_stages,
                }
            )

            done = T.alloc_var(T.bool)
            if fix_routing_mask_exists:
                is_fixed_routing = T.alloc_var(T.bool, init=False)
            else:
                is_fixed_routing = False
            if has_force_random and to_physical_map_exists:
                physical_experts_ready = T.alloc_var(T.bool, init=False)
                cached_num_physical_experts = T.alloc_var(T.int32, init=0)
            topk_idx_view = T.view(topk_idx, (num_tokens, num_physical_topk * 2), T.int32)

            if has_force_random and unmapped_topk_idx_exists and num_topk in (4, 6, 8):
                with T.SimdVF():
                    fill_init_mask_32b = S.pset(32, "PAT_ALL")
                    fill_init_neg_one = S.vdup(T.int32(-1), T.int32, fill_init_mask_32b)
                    S.vsts(unmapped_fill_i32_ub[0], fill_init_neg_one, fill_init_mask_32b)

            if has_bias or has_image_token_mask:
                with T.SimdVF():
                    T.clear(bias_ub)
                if has_bias:
                    T.copy(bias[:], bias_ub[:num_routed_experts])
            for work_id in T.Pipelined(
                T.ceildiv(num_tokens, num_cores),
                num_stages=num_stages,
                annotations={
                    "multi_buffer_eligible": [
                        logits_ub,
                        out_idx_ub,
                        out_weights_ub,
                    ]
                },
            ):
                row = work_id * num_cores + core_id
                if row < num_tokens:
                    done = False
                    if fix_routing_mask_exists:
                        is_fixed_routing = fix_routing_mask[row]
                    if has_image_token_mask:
                        if image_token_mask[row]:
                            T.copy(image_bias[:], bias_ub[:num_routed_experts])
                        elif has_bias:
                            T.copy(bias[:], bias_ub[:num_routed_experts])
                        else:
                            with T.SimdVF():
                                T.clear(bias_ub)

                    with T.SimdVF():
                        clear_mask_32b = S.pset(32, "PAT_ALL")
                        clear_zero_i32 = S.vdup(T.int32(0), T.int32, clear_mask_32b)
                        for clear_chunk in T.unroll(2, explicit=True):
                            S.vsts(out_idx_ub[clear_chunk * vector_length], clear_zero_i32, clear_mask_32b)
                        if has_image_token_mask or (
                            scoring_type == 3 and num_topk == 6 and not fix_routing_mask_exists and not mask_exists and not has_force_random
                        ):
                            clear_zero = S.vdup(T.float32(0), T.float32, clear_mask_32b)
                            S.vsts(out_weights_ub[0], clear_zero, clear_mask_32b)

                    # mask has the highest priority.
                    if mask_exists and not mask[row]:
                        with T.SimdVF():
                            masked_fill_mask_32b = S.pset(32, "PAT_ALL")
                            masked_fill_neg_one = S.vdup(T.int32(-1), T.int32, masked_fill_mask_32b)
                            for fill_chunk in T.unroll(2, explicit=True):
                                S.vsts(out_idx_ub[fill_chunk * vector_length], masked_fill_neg_one, masked_fill_mask_32b)
                            fill_zero = S.vdup(T.float32(0), T.float32, masked_fill_mask_32b)
                            S.vsts(out_weights_ub[0], fill_zero, masked_fill_mask_32b)
                        T.copy(out_idx_ub[: num_physical_topk * 2], topk_idx_view[row, :])
                        T.copy(out_weights_ub[:num_physical_topk], topk_weights[row, :])
                        if unmapped_topk_idx_exists:
                            if num_topk in (4, 8):
                                T.copy(out_idx_i64_ub[:num_topk], unmapped_topk_idx[row, :])
                            elif num_topk == 6:
                                T.copy(out_idx_i64_ub[:4], unmapped_topk_idx[row, :4])
                                for k in T.serial(4, num_topk):
                                    unmapped_topk_idx[row, k] = T.cast(-1, T.int64)
                            else:
                                for k in T.serial(num_topk):
                                    unmapped_topk_idx[row, k] = T.cast(-1, T.int64)
                        done = True

                    # force_random has priority over fix_routing. Generate the
                    # shrinking-range candidates in parallel, then use vector
                    # min reductions to lift collisions. This is the Ascend
                    # SIMD equivalent of the CUDA lane algorithm and produces
                    # a uniform sample without replacement.
                    if not done and has_force_random and force_random[row]:
                        num_physical_experts = T.alloc_var(T.int32, init=0)
                        if to_physical_map_exists:
                            if not physical_experts_ready:
                                for logical_idx in T.serial(num_logical_experts):
                                    cached_num_physical_experts = cached_num_physical_experts + logical_count[logical_idx]
                                physical_experts_ready = True
                            num_physical_experts = cached_num_physical_experts
                        else:
                            num_physical_experts = num_routed_experts + num_shared_experts

                        T.device_assert(num_physical_experts >= num_physical_topk, msg="not enough physical experts for force_random")

                        with T.SimtVF(threads=128):
                            tid = T.get_thread_binding()
                            if tid < vector_length:
                                if tid < num_physical_topk:
                                    T.rng_init(ep_rank, row * 32 + tid, 0)
                                    random_raw_ub[tid] = T.rng_rand() % (num_physical_experts - tid)
                                    out_weights_ub[tid] = T.rng_rand_float()
                                else:
                                    random_raw_ub[tid] = T.max_value(T.int32)

                        with T.SimdVF():
                            mask_32b = S.pset(32, "PAT_ALL")
                            lane_idx = S.vci(T.int32(0), T.int32)
                            int_max = S.vdup(T.max_value(T.int32), T.int32, mask_32b)
                            candidate_idx = S.alloc_var(T.int32)
                            resolved_idx = S.alloc_var(T.int32)
                            candidate_lane = S.alloc_var(T.int32)
                            candidate_idx = S.vld(random_raw_ub[0])
                            resolved_idx = int_max

                            for _ in T.unroll(max_physical_topk, explicit=True):
                                min_idx = S.vdupv(S.vcmin(candidate_idx, mask_32b), mask_32b)
                                active_lane = S.vcmp(candidate_idx, int_max, mask_32b, "lt")
                                candidate_lane = S.vsel(lane_idx, int_max, S.vcmp(candidate_idx, min_idx, mask_32b, "eq"))
                                candidate_lane = S.vsel(candidate_lane, int_max, active_lane)
                                min_lane = S.vdupv(S.vcmin(candidate_lane, mask_32b), mask_32b)
                                is_winner = S.vcmp(lane_idx, min_lane, mask_32b, "eq")
                                resolved_idx = S.vsel(candidate_idx, resolved_idx, is_winner)

                                incremented = S.vadds(candidate_idx, T.int32(1), mask_32b)
                                incremented_if_ge = S.vsel(incremented, candidate_idx, S.vcmp(candidate_idx, min_idx, mask_32b, "ge"))
                                candidate_idx = S.vsel(incremented_if_ge, candidate_idx, S.vcmp(min_lane, lane_idx, mask_32b, "lt"))
                                candidate_idx = S.vsel(candidate_idx, int_max, active_lane)
                                candidate_idx = S.vsel(int_max, candidate_idx, is_winner)

                            output_lane_mask = S.vcmp(lane_idx, S.vdup(num_physical_topk, T.int32, mask_32b), mask_32b, "lt")
                            output_offsets = T.reinterpret(S.vadd(lane_idx, lane_idx, mask_32b), "uint32x64")
                            S.vscatter(resolved_idx, out_idx_ub[0], output_offsets, output_lane_mask)

                        if unmapped_topk_idx_exists:
                            if num_topk in (4, 8):
                                T.copy(unmapped_fill_i64_ub[:num_topk], unmapped_topk_idx[row, :])
                            elif num_topk == 6:
                                T.copy(unmapped_fill_i64_ub[:4], unmapped_topk_idx[row, :4])
                                for k in T.serial(4, num_topk):
                                    unmapped_topk_idx[row, k] = T.cast(-1, T.int64)
                            else:
                                for k in T.serial(num_topk):
                                    unmapped_topk_idx[row, k] = T.cast(-1, T.int64)

                        T.copy(out_idx_ub[: num_physical_topk * 2], topk_idx_view[row, :])
                        T.copy(out_weights_ub[:num_physical_topk], topk_weights[row, :])
                        done = True

                    if not done:
                        T.copy(logits[row, :], logits_ub[:num_routed_experts])
                        # Scoring values without bias are retained for output
                        # weights. ranked_scores_ub contains the values used only
                        # for stable TopK selection.
                        if scoring_type == 2:
                            # SOFTMAX: global max and sum cover every expert
                            # group. Ranking follows the CUDA contract and uses
                            # raw logits plus bias.
                            with T.SimdVF():
                                scoring_mask_32b = S.pset(32, "PAT_ALL")
                                scoring_neg_inf = S.vdup(-T.infinity(T.float32), T.float32, scoring_mask_32b)
                                scoring_num_exp_vec = S.vdup(T.int32(num_routed_experts), T.int32, scoring_mask_32b)
                                max_acc = S.alloc_var(T.float32)
                                max_acc = scoring_neg_inf
                                for group_id in T.serial(num_groups):
                                    group_base = group_id * experts_per_group
                                    for vector_id in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = group_base + vector_id * vector_length
                                        indices = S.vci(T.int32(offset), T.int32)
                                        loaded_values = S.vld(logits_ub[offset])
                                        bounded_values = S.vsel(
                                            loaded_values, scoring_neg_inf, S.vcmp(indices, scoring_num_exp_vec, scoring_mask_32b, "lt")
                                        )
                                        max_acc = S.vmax(max_acc, bounded_values, scoring_mask_32b)
                                scoring_max_vec = S.vdupv(S.vcmax(max_acc, scoring_mask_32b), scoring_mask_32b)

                                for group_id in T.serial(num_groups):
                                    group_base = group_id * experts_per_group
                                    for vector_id in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = group_base + vector_id * vector_length
                                        indices = S.vci(T.int32(offset), T.int32)
                                        loaded_values = S.vld(logits_ub[offset])
                                        bounded_values = S.vsel(
                                            loaded_values, scoring_neg_inf, S.vcmp(indices, scoring_num_exp_vec, scoring_mask_32b, "lt")
                                        )
                                        exp_values = S.vexp(S.vsub(bounded_values, scoring_max_vec, scoring_mask_32b), scoring_mask_32b)
                                        S.vsts(scores_ub[offset], exp_values, scoring_mask_32b)

                                for group_id in T.serial(num_groups):
                                    group_base = group_id * experts_per_group
                                    for vector_id in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = group_base + vector_id * vector_length
                                        indices = S.vci(T.int32(offset), T.int32)
                                        valid = S.vcmp(indices, scoring_num_exp_vec, scoring_mask_32b, "lt")
                                        unbiased_ranked = S.vsel(S.vld(logits_ub[offset]), scoring_neg_inf, valid)
                                        if has_bias or has_image_token_mask:
                                            bias_values = S.vld(bias_ub[offset])
                                            ranked_values = S.vadd(unbiased_ranked, bias_values, scoring_mask_32b)
                                        else:
                                            ranked_values = unbiased_ranked
                                        S.vsts(ranked_scores_ub[offset], ranked_values, scoring_mask_32b)
                        else:
                            with T.SimdVF():
                                scoring_mask_32b = S.pset(32, "PAT_ALL")
                                scoring_neg_inf = S.vdup(-T.infinity(T.float32), T.float32, scoring_mask_32b)
                                zero_vec = S.vdup(0.0, T.float32, scoring_mask_32b)
                                one_vec = S.vdup(1.0, T.float32, scoring_mask_32b)
                                threshold_vec = S.vdup(20.0, T.float32, scoring_mask_32b)
                                if scoring_type == 1:
                                    sqrt_residual = S.alloc_var(T.float32)
                                    one_u32 = S.vdup(1, T.uint32, scoring_mask_32b)
                                    exponent_mask = S.vdup(0x7F800000, T.uint32, scoring_mask_32b)
                                    ulp_exponent_shift = S.vdup(23 << 23, T.uint32, scoring_mask_32b)
                                scoring_num_exp_vec = S.vdup(T.int32(num_routed_experts), T.int32, scoring_mask_32b)
                                for group_id in T.serial(num_groups):
                                    group_base = group_id * experts_per_group
                                    for vector_id in T.unroll(num_vregs_per_group, explicit=True):
                                        offset = group_base + vector_id * vector_length
                                        indices = S.vci(T.int32(offset), T.int32)
                                        valid = S.vcmp(indices, scoring_num_exp_vec, scoring_mask_32b, "lt")
                                        raw_values = S.vsel(S.vld(logits_ub[offset]), scoring_neg_inf, valid)

                                        if scoring_type == 0:
                                            scored_values = S.vdiv(
                                                one_vec,
                                                S.vadd(
                                                    one_vec,
                                                    S.vexp(S.vneg(raw_values, scoring_mask_32b), scoring_mask_32b),
                                                    scoring_mask_32b,
                                                ),
                                                scoring_mask_32b,
                                            )
                                        elif scoring_type == 1:
                                            softplus_log_values = S.vln(
                                                S.vadd(one_vec, S.vexp(raw_values, scoring_mask_32b), scoring_mask_32b), scoring_mask_32b
                                            )
                                            softplus_values = S.vsel(
                                                raw_values, softplus_log_values, S.vcmp(raw_values, threshold_vec, scoring_mask_32b, "gt")
                                            )
                                            sqrt_values = S.vsqrt(softplus_values, scoring_mask_32b)
                                            sqrt_bits = T.reinterpret(sqrt_values, "uint32x64")
                                            ulp_bits = S.vsub(
                                                S.vand(sqrt_bits, exponent_mask, scoring_mask_32b), ulp_exponent_shift, scoring_mask_32b
                                            )
                                            sqrt_ulp = T.reinterpret(ulp_bits, "float32x64")
                                            rounding_boundary = S.vmul(sqrt_values, sqrt_ulp, scoring_mask_32b)

                                            # Refine the hardware sqrt at FP32
                                            # rounding midpoints. vmadd computes
                                            # y*y-x as a fused residual; if the
                                            # positive residual exceeds y*ulp, y
                                            # is one representable value too high.
                                            sqrt_residual = sqrt_values
                                            S.vmadd(sqrt_residual, sqrt_values, S.vneg(softplus_values, scoring_mask_32b), scoring_mask_32b)
                                            previous_sqrt = T.reinterpret(S.vsub(sqrt_bits, one_u32, scoring_mask_32b), "float32x64")
                                            scored_values = S.vsel(
                                                previous_sqrt, sqrt_values, S.vcmp(sqrt_residual, rounding_boundary, scoring_mask_32b, "gt")
                                            )
                                        else:
                                            scored_values = raw_values

                                        scores = S.vsel(scored_values, zero_vec, valid)
                                        S.vsts(scores_ub[offset], scores, scoring_mask_32b)

                                        unbiased_ranked = S.vsel(scored_values, scoring_neg_inf, valid)
                                        if has_bias or has_image_token_mask:
                                            bias_values = S.vld(bias_ub[offset])
                                            ranked_values = S.vadd(unbiased_ranked, bias_values, scoring_mask_32b)
                                        else:
                                            ranked_values = unbiased_ranked
                                        S.vsts(ranked_scores_ub[offset], ranked_values, scoring_mask_32b)

                        # Stable TopK. Keep a single expert group in vector
                        # registers across all selections. Larger expert sets
                        # use the correctness-preserving multi-group scan.
                        with T.SimdVF():
                            topk_mask_32b = S.pset(32, "PAT_ALL")
                            topk_one_lane = S.pset(32, "PAT_VL1")
                            topk_neg_inf = S.vdup(-T.infinity(T.float32), T.float32, topk_mask_32b)
                            int_max = S.vdup(T.max_value(T.int32), T.int32, topk_mask_32b)
                            topk_num_exp_vec = S.vdup(T.int32(num_routed_experts), T.int32, topk_mask_32b)

                            scores_vec = S.alloc_local((num_vregs_per_group,), T.float32)
                            index_vec = S.alloc_local((num_vregs_per_group,), T.int32)
                            candidate_idx_vec = S.alloc_local((num_vregs_per_group,), T.int32)
                            max_acc = S.alloc_var(T.float32)
                            min_idx_acc = S.alloc_var(T.int32)

                            if num_groups == 1:
                                for vector_id in T.unroll(num_vregs_per_group, explicit=True):
                                    offset = vector_id * vector_length
                                    index_vec[vector_id] = S.vci(T.int32(offset), T.int32)
                                    scores_vec[vector_id] = S.vsel(
                                        S.vld(ranked_scores_ub[offset]),
                                        topk_neg_inf,
                                        S.vcmp(index_vec[vector_id], topk_num_exp_vec, topk_mask_32b, "lt"),
                                    )

                                if not is_fixed_routing:
                                    for k in T.serial(num_topk):
                                        # Merge score/index pairs together. The
                                        # previous implementation reduced all score
                                        # vregs first, then performed another full
                                        # cross-vreg min reduction for the matching
                                        # indices. Propagating the winning index with
                                        # each score merge removes that second tree.
                                        max_acc, min_idx_acc = _balanced_score_index_reduce(
                                            scores_vec, index_vec, num_vregs_per_group, topk_mask_32b
                                        )
                                        max_vec = S.vdupv(S.vcmax(max_acc, topk_mask_32b), topk_mask_32b)
                                        min_idx_acc = S.vsel(min_idx_acc, int_max, S.vcmp(max_acc, max_vec, topk_mask_32b, "eq"))
                                        winner_idx = S.vdupv(S.vcmin(min_idx_acc, topk_mask_32b), topk_mask_32b)
                                        S.vsts(out_idx_ub[2 * k], winner_idx, topk_one_lane, "ONEPT_B32")

                                        for vector_id in T.unroll(num_vregs_per_group, explicit=True):
                                            scores_vec[vector_id] = S.vsel(
                                                topk_neg_inf,
                                                scores_vec[vector_id],
                                                S.vcmp(index_vec[vector_id], winner_idx, topk_mask_32b, "eq"),
                                            )
                            else:
                                for k in T.serial(num_topk):
                                    max_acc = topk_neg_inf
                                    for group_id in T.serial(num_groups):
                                        group_base = group_id * experts_per_group
                                        for vector_id in T.unroll(num_vregs_per_group, explicit=True):
                                            offset = group_base + vector_id * vector_length
                                            index_vec[vector_id] = S.vci(T.int32(offset), T.int32)
                                            scores_vec[vector_id] = S.vsel(
                                                S.vld(ranked_scores_ub[offset]),
                                                topk_neg_inf,
                                                S.vcmp(index_vec[vector_id], topk_num_exp_vec, topk_mask_32b, "lt"),
                                            )
                                            max_acc = S.vmax(max_acc, scores_vec[vector_id], topk_mask_32b)
                                    max_vec = S.vdupv(S.vcmax(max_acc, topk_mask_32b), topk_mask_32b)

                                    min_idx_acc = int_max
                                    for group_id in T.serial(num_groups):
                                        group_base = group_id * experts_per_group
                                        for vector_id in T.unroll(num_vregs_per_group, explicit=True):
                                            offset = group_base + vector_id * vector_length
                                            index_vec[vector_id] = S.vci(T.int32(offset), T.int32)
                                            scores_vec[vector_id] = S.vsel(
                                                S.vld(ranked_scores_ub[offset]),
                                                topk_neg_inf,
                                                S.vcmp(index_vec[vector_id], topk_num_exp_vec, topk_mask_32b, "lt"),
                                            )
                                            candidate_idx_vec[vector_id] = S.vsel(
                                                index_vec[vector_id], int_max, S.vcmp(scores_vec[vector_id], max_vec, topk_mask_32b, "eq")
                                            )
                                            min_idx_acc = S.vmin(min_idx_acc, candidate_idx_vec[vector_id], topk_mask_32b)
                                    winner_idx = S.vdupv(S.vcmin(min_idx_acc, topk_mask_32b), topk_mask_32b)
                                    S.vsts(out_idx_ub[2 * k], winner_idx, topk_one_lane, "ONEPT_B32")

                                    for group_id in T.serial(num_groups):
                                        group_base = group_id * experts_per_group
                                        for vector_id in T.unroll(num_vregs_per_group, explicit=True):
                                            offset = group_base + vector_id * vector_length
                                            indices = S.vci(T.int32(offset), T.int32)
                                            ranked_values = S.vld(ranked_scores_ub[offset])
                                            masked_ranked_values = S.vsel(
                                                topk_neg_inf, ranked_values, S.vcmp(indices, winner_idx, topk_mask_32b, "eq")
                                            )
                                            S.vsts(ranked_scores_ub[offset], masked_ranked_values, topk_mask_32b)
                                    S.mem_bar("VST_VLD")

                        if fix_routing_mask_exists and is_fixed_routing:
                            if num_topk in (4, 8):
                                T.copy(unmapped_topk_idx[row, :], out_idx_i64_ub[:num_topk])
                            elif num_topk == 6:
                                T.copy(unmapped_topk_idx[row, :4], out_idx_i64_ub[:4])
                                for tail_id in T.serial(num_topk - 4):
                                    k = tail_id + 4
                                    out_idx_ub[2 * k] = T.cast(unmapped_topk_idx[row, k], T.int32)
                            else:
                                for k in T.serial(num_topk):
                                    out_idx_ub[2 * k] = T.cast(unmapped_topk_idx[row, k], T.int32)

                        with T.SimdVF():
                            output_mask_32b = S.pset(32, "PAT_ALL")
                            # CANN does not define PAT_VL6 and similar arbitrary
                            # predicate constants. Build a mask that supports
                            # every contracted TopK size (2/4/6/8) instead.
                            topk_mask = S.vcmp(
                                S.vci(T.int32(0), T.int32), S.vdup(T.int32(num_topk), T.int32, output_mask_32b), output_mask_32b, "lt"
                            )
                            selected_idx_i32, _ = S.vld2(out_idx_ub[0], "DINTLV_B32")
                            # out_idx_ub stores int64 indices as interleaved
                            # low/high int32 words. Expert indices are
                            # non-negative, so the deinterleaved low words are
                            # directly usable as vgather2 element offsets.
                            selected_idx_u32 = T.reinterpret(selected_idx_i32, "uint32x64")
                            weights_vec = S.vgather2(scores_ub[0], selected_idx_u32, topk_mask)
                            weight_sum = S.vdupv(S.vcadd(weights_vec, topk_mask), output_mask_32b, "POS_LOWEST")
                            weight_sum_eps = S.vadds(weight_sum, 1e-20, output_mask_32b)
                            normalized_weights = S.vmuls(S.vdiv(weights_vec, weight_sum_eps, topk_mask), routed_scaling_factor, topk_mask)
                            S.vsts(out_weights_ub[0], normalized_weights, topk_mask)

                        for shared_idx in T.serial(num_shared_experts):
                            output_slot = num_topk + shared_idx
                            out_idx_ub[2 * output_slot] = num_routed_experts + shared_idx
                            out_weights_ub[output_slot] = 1.0

                        if unmapped_topk_idx_exists:
                            if num_topk in (4, 8) and fix_routing_mask_exists and not mask_exists and not has_force_random:
                                T.copy(out_idx_i64_ub[:num_topk], unmapped_topk_idx[row, :])
                            elif not is_fixed_routing:
                                if num_topk in (4, 8):
                                    T.copy(out_idx_i64_ub[:num_topk], unmapped_topk_idx[row, :])
                                elif num_topk == 6:
                                    T.copy(out_idx_i64_ub[:4], unmapped_topk_idx[row, :4])
                                    for k in T.serial(4, num_topk):
                                        unmapped_topk_idx[row, k] = T.cast(out_idx_ub[2 * k], T.int64)
                                else:
                                    for k in T.serial(num_topk):
                                        unmapped_topk_idx[row, k] = T.cast(out_idx_ub[2 * k], T.int64)

                        T.copy(out_idx_ub[: num_physical_topk * 2], topk_idx_view[row, :])
                        T.copy(out_weights_ub[:num_physical_topk], topk_weights[row, :])

            if to_physical_map_exists:
                # A per-row SimtVF call is expensive in TileLang. Store the
                # logical indices first, then amortize one SIMT conversion
                # across every row owned by this core. Padding each row to the
                # static maximum output count keeps row/slot decoding cheap.
                map_rows_per_core = T.ceildiv(num_tokens, num_cores)
                map_tasks_per_core = map_rows_per_core * max_physical_topk
                with T.SimtVF(threads=physical_map_threads):
                    map_thread = T.get_thread_binding()
                    for map_iteration in T.serial(T.ceildiv(map_tasks_per_core, physical_map_threads)):
                        map_task = map_iteration * physical_map_threads + map_thread
                        if map_task < map_tasks_per_core:
                            map_work_id = map_task // max_physical_topk
                            output_slot = map_task % max_physical_topk
                            map_row = map_work_id * num_cores + core_id
                            if map_row < num_tokens and output_slot < num_physical_topk:
                                should_map = True
                                if has_force_random:
                                    should_map = not force_random[map_row]
                                if should_map:
                                    logical_idx = topk_idx_view[map_row, 2 * output_slot]
                                    if logical_idx >= 0:
                                        duplicate_count = logical_count[logical_idx]
                                        if duplicate_count == 1:
                                            topk_idx_view[map_row, 2 * output_slot] = to_physical_map[logical_idx, 0]
                                        else:
                                            duplicate_idx = (ep_rank + map_row * 23333) % duplicate_count
                                            topk_idx_view[map_row, 2 * output_slot] = to_physical_map[logical_idx, duplicate_idx]

    return moe_topk_gate_kernel_asc
