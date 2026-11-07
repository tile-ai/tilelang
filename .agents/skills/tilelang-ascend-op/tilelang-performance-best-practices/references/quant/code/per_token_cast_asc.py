# Ruff cannot model the TileLang macros retained after the shape-selection loop.
# They intentionally use the loop's final converged values.
# ruff: noqa: B023

import contextlib

import tilelang
from tilelang import language as T
from tilelang.language import simd as S

from tile_kernels.config import get_num_vec_cores
from tile_kernels.quant.common import CastInputConfig, CastOutputConfig, get_sf_shape


def _stage(level: int, manual: bool):
    return T.Stage(level) if manual else contextlib.nullcontext()


@T.macro
def _xor_u32(lhs, rhs):
    common = S.vand(lhs, rhs)
    return S.vsub(S.vadd(lhs, rhs), S.vshls(common, 1))


@T.macro
def _stochastic_round_vector(
    values,
    linear_base,
    quant_max,
    mantissa_bits,
    min_step_exp,
):
    abs_values = S.vmins(S.vabs(values), quant_max)
    abs_bits = T.reinterpret(abs_values, "uint32x64")
    exponent = T.reinterpret(
        S.vand(
            S.vshrs(abs_bits, 23),
            S.vdup(0xFF, T.uint32),
        ),
        "int32x64",
    )
    step_exponent = S.vmax(
        S.vadds(exponent, -mantissa_bits),
        S.vdup(min_step_exp + 127, T.int32),
    )
    step = T.reinterpret(
        S.vshls(T.reinterpret(step_exponent, "uint32x64"), 23),
        "float32x64",
    )

    lower_steps = S.vcvt(
        S.vdiv(abs_values, step),
        T.int32,
        round="ROUND_Z",
    )
    lower = S.vmul(S.vcvt(lower_steps, T.float32), step)
    upper = S.vmins(S.vadd(lower, step), quant_max)
    gap = S.vsub(upper, lower)
    has_gap = S.vcmps(gap, 0.0, op="gt")
    safe_gap = S.vsel(gap, S.vdup(1.0, T.float32), has_gap)
    probability = S.vdiv(S.vsub(abs_values, lower), safe_gap)

    random_seed = S.vadd(
        T.reinterpret(S.vci(T.int32(linear_base), T.int32), "uint32x64"),
        S.vdup(0x9E3779B9, T.uint32),
    )
    random_bits_1 = _xor_u32(random_seed, S.vshls(random_seed, 13))
    random_bits_2 = _xor_u32(random_bits_1, S.vshrs(random_bits_1, 17))
    random_bits_3 = _xor_u32(random_bits_2, S.vshls(random_bits_2, 5))
    random_mantissa = S.vor(
        S.vand(random_bits_3, S.vdup(0x007FFFFF, T.uint32)),
        S.vdup(0x3F800000, T.uint32),
    )
    random_uniform = S.vadds(
        T.reinterpret(random_mantissa, "float32x64"),
        -1.0,
    )
    rounded_abs = S.vsel(
        upper,
        lower,
        S.vcmp(random_uniform, probability, op="lt"),
    )
    sign = S.vand(
        T.reinterpret(values, "uint32x64"),
        S.vdup(0x80000000, T.uint32),
    )
    return T.reinterpret(
        S.vor(T.reinterpret(rounded_abs, "uint32x64"), sign),
        "float32x64",
    )


PREQUANT_ROWS = 64
TALL_PREQUANT_ROWS = 128
COL_MAJOR_PREQUANT_BLOCK_K = 512
TILE_COUNT_GRANULARITY = 256
PREQUANT_TILE_ELEMS = 16 * 1024

RAW_SPARSE_TILE_ELEMS = 16 * 1024

BF16_TILE_MAX_ROWS = 32
PACKED_PREQUANT_ROWS = 32

RAW_TILE_BYTES = 64 * 1024


def tall_prequant_tile_pays(num_tokens: int, hidden: int) -> bool:
    if num_tokens <= 0 or hidden % TILE_COUNT_GRANULARITY:
        return False
    tall_tiles = -(-num_tokens // TALL_PREQUANT_ROWS)
    rows_tall = tall_tiles * TALL_PREQUANT_ROWS
    rows_short = -(-num_tokens // PREQUANT_ROWS) * PREQUANT_ROWS
    if rows_tall != rows_short:
        return False
    hidden_tiles = hidden // TILE_COUNT_GRANULARITY
    return tall_tiles * hidden_tiles >= 2 * get_num_vec_cores()


def get_per_token_cast_kernel_asc(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool = False,
    cast_only: bool = False,
    stochastic_cast: bool = False,
    small_batch: bool = True,
):
    group_size = out_config.sf_block[1]

    assert out_config.dtype in (T.float8_e4m3fn, T.float4_e2m1fn)
    assert out_config.sf_block[0] == 1
    assert not out_config.use_packed_ue8m0 or out_config.round_sf
    assert not (sf_only and cast_only), "SF-only and cast-only modes are mutually exclusive"
    if in_config.with_sf:
        assert in_config.dtype in (T.float8_e4m3fn, T.float4_e2m1fn)
        assert in_config.sf_block[0] in (1, 32, 128)
        assert in_config.sf_block[1] in (32, 128)
        assert not sf_only and not cast_only, "pre-quantized input only supports full cast"
    else:
        assert in_config.dtype in (T.float32, T.bfloat16)

    if prequant_row_major_path_ok(hidden, in_config, out_config, sf_only, cast_only, stochastic_cast, small_batch):
        return _get_prequant_row_major_cast_kernel(
            hidden=hidden,
            token_stride=token_stride,
            in_config=in_config,
            out_config=out_config,
        )

    if group_size in (16, 32, 64, 128):
        return _get_grouped_cast_kernel(
            hidden=hidden,
            token_stride=token_stride,
            in_config=in_config,
            out_config=out_config,
            sf_only=sf_only,
            cast_only=cast_only,
            stochastic_cast=stochastic_cast,
            small_batch=small_batch,
        )

    assert not in_config.with_sf, "pre-quantized input requires grouped output scales"
    assert group_size == hidden and hidden % 64 == 0
    return _get_full_row_cast_kernel(
        hidden=hidden,
        token_stride=token_stride,
        input_dtype=in_config.dtype,
        out_config=out_config,
        sf_only=sf_only,
        cast_only=cast_only,
        stochastic_cast=stochastic_cast,
    )


@tilelang.jit
def _get_grouped_cast_kernel(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
    sf_only: bool,
    cast_only: bool,
    stochastic_cast: bool,
    small_batch: bool = True,
):
    group_size = out_config.sf_block[1]
    assert group_size in (16, 32, 64, 128)
    input_dtype = in_config.dtype
    has_input_sf = in_config.with_sf
    if has_input_sf:
        assert input_dtype in (T.float8_e4m3fn, T.float4_e2m1fn)
        input_block_m, input_group_size = in_config.sf_block
    else:
        assert input_dtype in (T.float32, T.bfloat16)
        input_block_m, input_group_size = 1, 1

    num_cores = get_num_vec_cores()
    raw_row_major = not has_input_sf and not out_config.use_tma_aligned_col_major_sf and not out_config.use_packed_ue8m0
    raw_packed_row_major = not has_input_sf and not out_config.use_tma_aligned_col_major_sf and out_config.use_packed_ue8m0
    raw_col_major = not has_input_sf and out_config.use_tma_aligned_col_major_sf
    wide_group_block_k = group_size * 8
    use_wide_group_tile = raw_row_major and group_size >= 64 and hidden >= wide_group_block_k
    packed_row_block_k = group_size * 32
    use_packed_row_tile = raw_packed_row_major and hidden >= packed_row_block_k
    col_major_block_k = max(
        128,
        group_size * (2 if out_config.use_packed_ue8m0 else 1),
    )
    _col_major_out_row_bytes = col_major_block_k // 2 if out_config.dtype == T.float4_e2m1fn else col_major_block_k * out_config.dtype.bytes
    if _col_major_out_row_bytes < 128:
        col_major_block_k *= 128 // _col_major_out_row_bytes
    use_col_major_tile = raw_col_major and hidden >= col_major_block_k
    tall_prequant_rows = TALL_PREQUANT_ROWS
    tall_prequant_staging_is_bf16 = (
        out_config.dtype == T.float4_e2m1fn
        and out_config.use_packed_ue8m0
        and out_config.round_sf
        and in_config.use_packed_ue8m0
        and group_size == 32
        and input_group_size >= 2 * group_size
        and input_group_size % group_size == 0
        and (tall_prequant_rows * 2) % 64 == 0
        and not sf_only
        and not cast_only
        and not stochastic_cast
    )
    use_tall_prequant_tile = (
        has_input_sf
        and input_dtype in (T.float8_e4m3fn, T.float4_e2m1fn)
        and out_config.use_tma_aligned_col_major_sf
        and tall_prequant_staging_is_bf16
        and input_block_m % tall_prequant_rows == 0
        and not small_batch
        and hidden % COL_MAJOR_PREQUANT_BLOCK_K == 0
    )
    _staging_bytes = 4
    _wide_k_ok = True
    _wide_k_settled = False
    for _ in range(8):
        if out_config.use_tma_aligned_col_major_sf:
            _budget = 2 * PREQUANT_TILE_ELEMS if use_tall_prequant_tile else PREQUANT_TILE_ELEMS
            prequant_block_k = (
                COL_MAJOR_PREQUANT_BLOCK_K
                if (
                    _wide_k_ok
                    and min(input_block_m, _budget // COL_MAJOR_PREQUANT_BLOCK_K) >= min(input_block_m, _budget // 256)
                    and hidden // COL_MAJOR_PREQUANT_BLOCK_K >= 8
                )
                else 256
            )
        elif out_config.use_packed_ue8m0:
            narrow_block_k = PREQUANT_TILE_ELEMS // PACKED_PREQUANT_ROWS
            prequant_block_k = narrow_block_k if hidden % narrow_block_k == 0 else packed_row_block_k
        elif group_size >= 64:
            _row_narrow_k = PREQUANT_TILE_ELEMS // PACKED_PREQUANT_ROWS
            prequant_block_k = _row_narrow_k if hidden % _row_narrow_k == 0 else wide_group_block_k
        else:
            prequant_block_k = 256
        use_prequant_tile = has_input_sf and input_block_m > 1 and hidden >= prequant_block_k
        if use_prequant_tile:
            block_k = prequant_block_k
        elif use_col_major_tile:
            block_k = col_major_block_k
        elif use_packed_row_tile:
            block_k = packed_row_block_k
        elif use_wide_group_tile:
            block_k = wide_group_block_k
        else:
            block_k = 256
        groups_per_tile = block_k // group_size
        dense_row_tile = (raw_row_major or use_packed_row_tile) and groups_per_tile * out_config.sf_dtype.bytes >= 32
        tall_prequant_single_stage = use_tall_prequant_tile and input_dtype == T.float4_e2m1fn
        prequant_double_buffer = use_prequant_tile and hidden % block_k == 0 and not tall_prequant_single_stage
        num_stages = 2 if (not has_input_sf and (dense_row_tile or use_col_major_tile)) or prequant_double_buffer else 1
        if use_prequant_tile:
            _prequant_rows_stay = not small_batch and (
                (
                    not out_config.use_tma_aligned_col_major_sf
                    and (
                        out_config.use_packed_ue8m0
                        or (group_size >= 64 and prequant_block_k == PREQUANT_TILE_ELEMS // PACKED_PREQUANT_ROWS)
                    )
                )
                or (
                    out_config.use_tma_aligned_col_major_sf
                    and prequant_block_k == 256
                    and input_block_m > PREQUANT_TILE_ELEMS // prequant_block_k
                )
            )
            prequant_tile_elems = 2 * PREQUANT_TILE_ELEMS if use_tall_prequant_tile else PREQUANT_TILE_ELEMS
            if _prequant_rows_stay:
                prequant_tile_elems = max(prequant_tile_elems, RAW_TILE_BYTES // _staging_bytes)
            max_prequant_rows = min(input_block_m, prequant_tile_elems // block_k)
            block_m = 1
            for candidate_rows in (128, 64, 32, 16, 8, 4, 2):
                if candidate_rows <= max_prequant_rows and input_block_m % candidate_rows == 0:
                    block_m = candidate_rows
                    break
            num_sf_slots = block_m * groups_per_tile
        elif use_col_major_tile:
            col_major_tile_elems = RAW_TILE_BYTES // input_dtype.bytes if not has_input_sf and not small_batch else 16 * 1024
            max_col_major_rows = col_major_tile_elems // block_k
            rows_per_sf_batch = 64 // groups_per_tile
            block_m = max_col_major_rows // rows_per_sf_batch * rows_per_sf_batch
            num_sf_slots = block_m * groups_per_tile
        else:
            if dense_row_tile:
                dense_row_slots = 512 * 4 // input_dtype.bytes if not has_input_sf and not small_batch else 512
                num_sf_slots = min(dense_row_slots, dense_row_slots * 32 // group_size)
                block_m = num_sf_slots // groups_per_tile
            else:
                raw_sparse_cap = BF16_TILE_MAX_ROWS if has_input_sf else RAW_SPARSE_TILE_ELEMS
                block_m = 1 if group_size == 32 or hidden % block_k else max(1, min(raw_sparse_cap, RAW_SPARSE_TILE_ELEMS // block_k))
                num_sf_slots = 64 if block_m == 1 else block_m * groups_per_tile

        _sf_rf = 1
        _sf_old_branch = out_config.dtype == T.float4_e2m1fn and group_size == 32
        if (
            not out_config.use_tma_aligned_col_major_sf
            and not out_config.use_packed_ue8m0
            and (has_input_sf or (not stochastic_cast and (_sf_old_branch or input_dtype != T.bfloat16)))
        ):
            _sf_min_rows = (16 if input_dtype != T.bfloat16 else 32) if not has_input_sf else (16 if input_group_size <= group_size else 32)
            _sf_need = groups_per_tile * out_config.sf_dtype.bytes
            if _sf_need < 128:
                _sf_rf = 1
                while _sf_rf * 2 <= 128 // _sf_need and (
                    block_k * _sf_rf * 2 <= hidden
                    and hidden % (block_k * _sf_rf * 2) == 0
                    and block_m % (_sf_rf * 2) == 0
                    and block_m // (_sf_rf * 2) >= _sf_min_rows
                ):
                    _sf_rf *= 2
                if (not has_input_sf and not _sf_old_branch and input_dtype != T.bfloat16 and _sf_rf * _sf_need < 128) or (
                    has_input_sf and group_size >= 64 and _sf_rf * _sf_need < 128
                ):
                    _sf_rf = 1
        if _sf_rf > 1:
            block_k *= _sf_rf
            block_m //= _sf_rf
            groups_per_tile = block_k // group_size
            num_sf_slots = block_m * groups_per_tile

        _rf = 1 if (has_input_sf or _sf_rf > 1) else 8

        while _rf > 1 and (block_k * _rf > hidden or hidden % (block_k * _rf) or block_m % _rf or block_m // _rf > 4):
            _rf //= 2

        if _rf > 1:
            block_k *= _rf

            block_m //= _rf

            groups_per_tile = block_k // group_size

            num_sf_slots = block_m * groups_per_tile

        use_packed_row_micro_loop = use_packed_row_tile and group_size >= 32 and block_k > 256
        packed_row_micro_k = 256
        quantize_bf16_micro = 2
        quantize_micro_vectors = 8 if block_k > 1024 else 16
        packed_row_micro_vectors = packed_row_micro_k // 64
        packed_row_micro_tiles = block_k // packed_row_micro_k

        is_bf16_input = input_dtype == T.bfloat16
        is_fp4_input = input_dtype == T.float4_e2m1fn
        is_fp4 = out_config.dtype == T.float4_e2m1fn
        is_packed_sf = out_config.use_packed_ue8m0
        is_col_major_sf = out_config.use_tma_aligned_col_major_sf

        _three_stages_ok = (
            has_input_sf
            and out_config.round_sf
            and group_size == 32
            and not is_fp4_input
            and not is_fp4
            and prequant_double_buffer
            and block_m >= 32
            and not small_batch
            and not stochastic_cast
            and not cast_only
            and not sf_only
        )
        num_stages = 3 if (_three_stages_ok and num_stages == 2) else num_stages
        pipeline_offset_annotations = {"enable_offset": True} if not small_batch else None
        manual_pipeline_stages = pipeline_offset_annotations is not None and num_stages >= 2
        sf_ub_dtype = T.uint8 if is_packed_sf else T.float32
        sf_storage_words = (num_sf_slots * sf_ub_dtype.bytes + 3) // 4
        sf_inv_offset_words = num_stages * sf_storage_words
        sf_workspace_words = sf_inv_offset_words + num_sf_slots
        if is_col_major_sf:
            sf_ub_shape = (T.ceildiv(groups_per_tile, 2), block_m * 2) if is_packed_sf else (groups_per_tile, block_m)
        else:
            sf_ub_shape = (block_m, groups_per_tile)
        quant_max = 6.0 if is_fp4 else 448.0
        num_tokens = T.dynamic("num_tokens")
        _one = T.ceildiv(num_tokens, T.max(num_tokens, 1))

        def _lim(n):
            return T.serial(n * _one) if _rf > 1 else T.unroll(n, explicit=True)

        out_sf_stride = T.dynamic("out_sf_stride")
        x_sf_shape = get_sf_shape((num_tokens, hidden), in_config) if has_input_sf else (1, 1)
        sf_shape = get_sf_shape((num_tokens, hidden), out_config)
        num_hidden_tiles = T.ceildiv(hidden, block_k)

        def sf_slot(row, group):
            if is_col_major_sf:
                if is_packed_sf:
                    return group // 2 * (block_m * 2) + row * 2 + group % 2
                return group * block_m + row
            return row * groups_per_tile + group

        def sf_inv_slot(slot):
            return sf_inv_offset_words + slot

        if has_input_sf:
            input_groups_per_tile = block_k // input_group_size
            input_sf_value_slots = (input_groups_per_tile + 63) // 64 * 64
            input_sf_raw_dtype = T.uint8 if in_config.use_packed_ue8m0 else T.float32
            input_sf_col_major = in_config.use_tma_aligned_col_major_sf
            if input_sf_col_major:
                input_sf_raw_shape = (
                    (max(128, (input_groups_per_tile + 1) // 2), 2) if in_config.use_packed_ue8m0 else (max(64, input_groups_per_tile), 1)
                )
            else:
                input_sf_raw_shape = (
                    (1, max(256, input_groups_per_tile)) if in_config.use_packed_ue8m0 else (1, max(64, input_groups_per_tile))
                )
            input_sf_raw_slots = input_sf_raw_shape[0] * input_sf_raw_shape[1]
        _tile_bf16_unfused = (
            not stochastic_cast
            and not cast_only
            and not sf_only
            and (is_fp4_input or is_fp4)
            and has_input_sf
            and in_config.use_packed_ue8m0
            and (not is_fp4_input or is_fp4)
            and group_size >= 32
            and (not is_fp4_input or input_group_size in (32, 128))
            and block_m <= BF16_TILE_MAX_ROWS
        )

        _input_group_ratio_pow2 = (
            input_group_size % group_size == 0 and (input_group_size // group_size) & (input_group_size // group_size - 1) == 0
        )
        if is_col_major_sf and is_packed_sf:
            _col_batch_in_one_group = (block_m * 2) % 64 == 0
            _batch_input_scale_ok = _input_group_ratio_pow2
            _input_group_needed = 2 * group_size
        elif is_col_major_sf:
            _col_batch_in_one_group = block_m % 64 == 0
            _batch_input_scale_ok = _input_group_ratio_pow2
            _input_group_needed = group_size
        else:
            _col_batch_in_one_group = True
            _input_group_ratio = input_group_size // group_size
            _batch_input_scale_ok = (
                64 % groups_per_tile == 0
                and _input_group_ratio & (_input_group_ratio - 1) == 0
                and (input_group_size < 64 or group_size == 32 or not _tile_bf16_unfused)
            )
            _input_group_needed = group_size

        fast_reduce = (
            group_size == 32
            and (block_k % 256 == 0 or (block_k == 128 and block_m % 2 == 0 and block_m >= 32))
            and block_m > 1
            and (not is_col_major_sf or is_packed_sf or block_m >= 64)
        )

        fuse_input_scale = (
            has_input_sf
            and out_config.round_sf
            and (not is_fp4_input or is_fp4 or fast_reduce)
            and _batch_input_scale_ok
            and input_group_size >= _input_group_needed
            and input_group_size % group_size == 0
        )

        fuse_input_scale_per_lane = fuse_input_scale and not is_col_major_sf
        fuse_input_scale_col_lanes = fuse_input_scale and is_col_major_sf and not _col_batch_in_one_group

        @T.macro
        def make_input_scale_lanes(input_sf_values_ub):
            groups = S.vand(
                S.vci(0, T.int32),
                S.vdup(groups_per_tile - 1, T.int32),
            )
            shift = (input_group_size // group_size).bit_length() - 1
            index = S.vshrs(groups, shift) if shift else groups
            return S.vgather2(
                input_sf_values_ub[0],
                T.reinterpret(index, "uint32x64"),
            )

        @T.macro
        def tile_input_scale(input_sf_values_ub, sf_batch, input_scale_lanes):
            if fuse_input_scale_per_lane:
                return input_scale_lanes
            if fuse_input_scale_col_lanes:
                slot = S.vadds(S.vci(0, T.int32), sf_batch * 64)
                _ratio_shift = (input_group_size // group_size).bit_length() - 1
                if is_packed_sf:
                    pair = S.vshrs(slot, (block_m * 2).bit_length() - 1)
                    out_group = S.vor(
                        S.vshls(pair, 1),
                        S.vand(slot, S.vdup(1, T.int32)),
                    )
                else:
                    out_group = S.vshrs(slot, block_m.bit_length() - 1)
                in_group = S.vshrs(out_group, _ratio_shift) if _ratio_shift else out_group
                return S.vgather2(
                    input_sf_values_ub[0],
                    T.reinterpret(in_group, "uint32x64"),
                )
            if is_packed_sf:
                out_group = sf_batch * 64 // (block_m * 2) * 2
            else:
                out_group = sf_batch * 64 // block_m
            in_group = out_group * group_size // input_group_size
            return S.vld(input_sf_values_ub[in_group], dist="BRC_B32")

        @T.macro
        def clear_raw_input(x_ub):
            with T.SimdVF():
                zero = S.vdup(0.0, input_dtype)
                for row in T.serial(block_m):
                    for vector in T.unroll(
                        block_k // (128 if is_bf16_input else 64),
                        explicit=True,
                    ):
                        S.vsts(x_ub[row, vector * (128 if is_bf16_input else 64)], zero)

        @T.macro
        def clear_quantized_input(x_ub_uint8):
            with T.SimdVF():
                mask = S.pset(8, "PAT_VL128" if is_fp4_input else "PAT_ALL")
                vector_bytes = 128 if is_fp4_input else 256
                physical_block_k = block_k // 2 if is_fp4_input else block_k
                for row in T.serial(block_m):
                    for vector in T.unroll(
                        physical_block_k // vector_bytes,
                        explicit=True,
                    ):
                        S.vsts(
                            x_ub_uint8[row, vector * vector_bytes],
                            S.vdup(0, T.uint8),
                            mask=mask,
                            extent=vector_bytes,
                        )

        @T.macro
        def clear_input_scales(input_sf_raw_linear_ub):
            with T.SimdVF():
                if in_config.use_packed_ue8m0:
                    S.vsts(
                        input_sf_raw_linear_ub[0],
                        S.vdup(0, input_sf_raw_dtype),
                    )
                else:
                    for sf_batch in T.unroll(
                        input_sf_raw_slots // 64,
                        explicit=True,
                    ):
                        S.vsts(
                            input_sf_raw_linear_ub[sf_batch * 64],
                            S.vdup(0.0, input_sf_raw_dtype),
                        )

        @T.macro
        def load_input_scale(input_sf_values_ub, vector):
            if input_group_size == 32:
                mask_low = S.pset(32, "PAT_VL32")
                scale_low = S.vld(
                    input_sf_values_ub[vector * 2],
                    dist="BRC_B32",
                )
                scale_high = S.vld(
                    input_sf_values_ub[vector * 2 + 1],
                    dist="BRC_B32",
                )
                return S.vsel(scale_low, scale_high, mask_low)
            return S.vld(
                input_sf_values_ub[vector // 2],
                dist="BRC_B32",
            )

        bf16_tile = (
            not stochastic_cast
            and not cast_only
            and not sf_only
            and (is_fp4_input or is_fp4 or group_size == 32)
            and (
                (fuse_input_scale and group_size >= 32)
                or _tile_bf16_unfused
                or (
                    is_fp4
                    and has_input_sf
                    and in_config.use_packed_ue8m0
                    and group_size >= 32
                    and (not is_fp4_input or input_group_size in (32, 128))
                    and not fuse_input_scale
                )
            )
        )
        staging_dtype = T.bfloat16 if bf16_tile else T.float32
        _next_staging_bytes = 2 if bf16_tile else 4
        if _next_staging_bytes != _staging_bytes:
            _staging_bytes = _next_staging_bytes
            _redo = True
        elif _wide_k_ok and prequant_block_k == COL_MAJOR_PREQUANT_BLOCK_K and use_prequant_tile and num_stages == 3:
            _wide_k_ok, _staging_bytes, _redo = False, 4, True
        else:
            _redo = False
        if not _redo:
            _wide_k_settled = True
            break

    assert _wide_k_settled, "block shape selection did not converge within 8 passes"

    fuse_amax_into_dequant = (
        has_input_sf
        and not is_fp4_input
        and not bf16_tile
        and group_size == 128
        and not fast_reduce
        and block_m <= BF16_TILE_MAX_ROWS
        and not cast_only
        and not sf_only
        and not stochastic_cast
    )

    input_scale_bf16 = bf16_tile and is_fp4_input and not fuse_input_scale and input_group_size in (32, 128)

    reduce_reads_bf16 = bf16_tile if has_input_sf else is_bf16_input

    no_dequant_tile = (
        has_input_sf
        and fuse_input_scale
        and not is_fp4_input
        and not is_fp4
        and bf16_tile
        and fast_reduce
        and num_stages in (2, 3)
        and not cast_only
        and not sf_only
        and not stochastic_cast
    )

    reduce_reads_fp8 = no_dequant_tile

    bf16_inverse = (
        not no_dequant_tile
        and reduce_reads_bf16
        and is_fp4
        and out_config.round_sf
        and group_size >= 32
        and not stochastic_cast
        and not cast_only
        and not sf_only
    )

    @T.macro
    def load_as_fp32(x_ub, row, col):
        if no_dequant_tile:
            return S.vcvt(S.vld(x_ub[row, col], dist="UNPK4_B8"), T.float32)
        if has_input_sf:
            if bf16_tile:
                return S.vcvt(
                    S.vld(x_ub[row, col], dist="UNPK_B16"),
                    T.float32,
                    part=0,
                )
            return S.vld(x_ub[row, col])
        if is_bf16_input:
            return S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"), T.float32, part=0)
        return S.vld(x_ub[row, col])

    amax_row_major = (
        fast_reduce
        and (block_m >= 32 or not is_col_major_sf)
        and not (
            is_col_major_sf and is_packed_sf and block_k >= 256 and not stochastic_cast and block_m <= 32 and not (is_fp4_input and is_fp4)
        )
    )

    amax_needs_gather = amax_row_major and is_col_major_sf

    bf16_amax = amax_row_major and bf16_tile and is_col_major_sf and groups_per_tile == 8 and num_stages == 1
    amax_dtype = T.bfloat16 if bf16_amax else T.float32
    amax_row_pitch = 16 if bf16_amax else groups_per_tile
    amax_slots = block_m * amax_row_pitch if bf16_amax else num_sf_slots

    row_major_inverse = (
        bf16_inverse
        and group_size == 32
        and (sf_inv_offset_words * 2) % 8 == 0
        and (
            (is_col_major_sf and is_packed_sf and groups_per_tile == 8 and block_m % 32 == 0)
            or (not is_col_major_sf and groups_per_tile % 4 == 0)
        )
    )

    reorder_inverse_rows = row_major_inverse and is_col_major_sf

    def sf_inv_bf16_slot(slot):
        return sf_inv_offset_words * 2 + slot * 2

    @T.macro
    def decode_input_scales(input_sf_raw_linear_ub, input_sf_values_ub):
        with T.SimdVF():
            for sf_batch in _lim(input_sf_value_slots // 64):
                sf_offset = sf_batch * 64
                if in_config.use_packed_ue8m0:
                    exponents = T.reinterpret(
                        S.vld(
                            input_sf_raw_linear_ub[sf_offset],
                            dist="UNPK4_B8",
                        ),
                        "uint32x64",
                    )
                    if input_scale_bf16:
                        words = S.vor(
                            S.vshls(exponents, 7),
                            S.vshls(exponents, 23),
                        )
                        scales = T.reinterpret(words, "float32x64")
                    else:
                        scales = T.reinterpret(
                            S.vshls(exponents, 23),
                            "float32x64",
                        )
                else:
                    scales = S.vld(input_sf_raw_linear_ub[sf_offset])
                S.vsts(input_sf_values_ub[sf_offset], scales)

    @T.macro
    def apply_input_scale(values, input_sf_values_ub, vector):
        if fuse_input_scale:
            return values
        return S.vmul(values, load_input_scale(input_sf_values_ub, vector))

    def input_scale_bf16_slot(group):
        return group * 2

    @T.macro
    def dequantize_input(x_ub, input_sf_values_ub, input_sf_bf16_ub, dequantized_ub, amax_ub):
        with T.SimdVF():
            if is_fp4_input and bf16_tile and fuse_input_scale:
                for row in T.serial(block_m):
                    for pair in _lim(block_k // 128):
                        col = pair * 128
                        x_fp4 = S.vld(x_ub[row, col], dist="UNPK4_B8")
                        S.vsts(dequantized_ub[row, col], S.vcvt(x_fp4, T.bfloat16))
            elif is_fp4_input and bf16_tile and input_group_size == 128:
                for pair in _lim(block_k // 128):
                    col = pair * 128
                    scale = T.reinterpret(
                        S.vld(
                            input_sf_bf16_ub[input_scale_bf16_slot(col // input_group_size)],
                            dist="BRC_B32",
                        ),
                        "bfloat16x128",
                    )
                    for row in T.serial(block_m):
                        x_fp4 = S.vld(x_ub[row, col], dist="UNPK4_B8")
                        S.vsts(
                            dequantized_ub[row, col],
                            S.vmul(S.vcvt(x_fp4, T.bfloat16), scale),
                        )
            elif is_fp4_input and bf16_tile:
                lanes = S.vci(0, T.int16)
                low_quarter = S.vcmps(lanes, 32, op="lt")
                low_half = S.vcmps(lanes, 64, op="lt")
                low_three_quarters = S.vcmps(lanes, 96, op="lt")
                for pair in _lim(block_k // 128):
                    col = pair * 128
                    group = col // input_group_size
                    first = T.reinterpret(
                        S.vld(
                            input_sf_bf16_ub[input_scale_bf16_slot(group)],
                            dist="BRC_B32",
                        ),
                        "bfloat16x128",
                    )
                    second = T.reinterpret(
                        S.vld(
                            input_sf_bf16_ub[input_scale_bf16_slot(group + 1)],
                            dist="BRC_B32",
                        ),
                        "bfloat16x128",
                    )
                    third = T.reinterpret(
                        S.vld(
                            input_sf_bf16_ub[input_scale_bf16_slot(group + 2)],
                            dist="BRC_B32",
                        ),
                        "bfloat16x128",
                    )
                    fourth = T.reinterpret(
                        S.vld(
                            input_sf_bf16_ub[input_scale_bf16_slot(group + 3)],
                            dist="BRC_B32",
                        ),
                        "bfloat16x128",
                    )
                    scale = S.vsel(
                        S.vsel(first, second, low_quarter),
                        S.vsel(third, fourth, low_three_quarters),
                        low_half,
                    )
                    for row in T.serial(block_m):
                        x_fp4 = S.vld(x_ub[row, col], dist="UNPK4_B8")
                        S.vsts(
                            dequantized_ub[row, col],
                            S.vmul(S.vcvt(x_fp4, T.bfloat16), scale),
                        )
            elif is_fp4_input:
                if not fuse_input_scale:
                    zero_bf16 = S.vdup(0.0, T.bfloat16)
                    for pair in _lim(block_k // 128):
                        col = pair * 128
                        scale_low = load_input_scale(input_sf_values_ub, pair * 2)
                        scale_high = load_input_scale(input_sf_values_ub, pair * 2 + 1)
                        for row in T.serial(block_m):
                            x_fp4 = S.vld(x_ub[row, col], dist="UNPK4_B8")
                            x_bf16 = S.vcvt(x_fp4, T.bfloat16)
                            x_uint_low, x_uint_high = S.vintlv(zero_bf16, x_bf16)
                            x_low = T.reinterpret(x_uint_low, "float32x64")
                            x_high = T.reinterpret(x_uint_high, "float32x64")
                            S.vsts(dequantized_ub[row, col], S.vmul(x_low, scale_low))
                            S.vsts(dequantized_ub[row, col + 64], S.vmul(x_high, scale_high))
                else:
                    for row in T.serial(block_m):
                        zero_bf16 = S.vdup(0.0, T.bfloat16)
                        for pair in _lim(block_k // 128):
                            col = pair * 128
                            x_fp4 = S.vld(x_ub[row, col], dist="UNPK4_B8")
                            x_bf16 = S.vcvt(x_fp4, T.bfloat16)
                            x_uint_low, x_uint_high = S.vintlv(zero_bf16, x_bf16)
                            x_low = T.reinterpret(x_uint_low, "float32x64")
                            x_high = T.reinterpret(x_uint_high, "float32x64")
                            S.vsts(
                                dequantized_ub[row, col],
                                apply_input_scale(
                                    x_low,
                                    input_sf_values_ub,
                                    pair * 2,
                                ),
                            )
                            S.vsts(
                                dequantized_ub[row, col + 64],
                                apply_input_scale(
                                    x_high,
                                    input_sf_values_ub,
                                    pair * 2 + 1,
                                ),
                            )
            elif fuse_amax_into_dequant:
                if not fuse_input_scale:
                    for group in _lim(groups_per_tile):
                        col = group * 128
                        scale_low = load_input_scale(input_sf_values_ub, group * 2)
                        scale_high = load_input_scale(input_sf_values_ub, group * 2 + 1)
                        for row in T.serial(block_m):
                            low = S.vmul(S.vcvt(S.vld(x_ub[row, col], dist="UNPK4_B8"), T.float32), scale_low)
                            high = S.vmul(S.vcvt(S.vld(x_ub[row, col + 64], dist="UNPK4_B8"), T.float32), scale_high)
                            S.vsts(dequantized_ub[row, col], low)
                            S.vsts(dequantized_ub[row, col + 64], high)
                            S.vsts(
                                amax_ub[sf_slot(row, group)],
                                S.vcmax(S.vmax(S.vabs(low), S.vabs(high))),
                                dist="ONEPT_B32",
                            )
                else:
                    for row in T.serial(block_m):
                        for group in _lim(groups_per_tile):
                            col = group * 128
                            low = apply_input_scale(
                                S.vcvt(S.vld(x_ub[row, col], dist="UNPK4_B8"), T.float32),
                                input_sf_values_ub,
                                group * 2,
                            )
                            high = apply_input_scale(
                                S.vcvt(S.vld(x_ub[row, col + 64], dist="UNPK4_B8"), T.float32),
                                input_sf_values_ub,
                                group * 2 + 1,
                            )
                            S.vsts(dequantized_ub[row, col], low)
                            S.vsts(dequantized_ub[row, col + 64], high)
                            S.vsts(
                                amax_ub[sf_slot(row, group)],
                                S.vcmax(S.vmax(S.vabs(low), S.vabs(high))),
                                dist="ONEPT_B32",
                            )
            else:
                if not fuse_input_scale:
                    for vector in _lim(block_k // 64):
                        col = vector * 64
                        scale = load_input_scale(input_sf_values_ub, vector)
                        for row in T.serial(block_m):
                            x_fp8 = S.vld(x_ub[row, col], dist="UNPK4_B8")
                            widened = S.vmul(S.vcvt(x_fp8, T.float32), scale)
                            if bf16_tile:
                                S.vsts(
                                    dequantized_ub[row, col],
                                    S.vshrs(T.reinterpret(widened, "uint32x64"), 16),
                                    dist="PK_B32",
                                )
                            else:
                                S.vsts(dequantized_ub[row, col], widened)
                else:
                    for row in T.serial(block_m):
                        for vector in _lim(block_k // 64):
                            col = vector * 64
                            x_fp8 = S.vld(x_ub[row, col], dist="UNPK4_B8")
                            widened = apply_input_scale(
                                S.vcvt(x_fp8, T.float32),
                                input_sf_values_ub,
                                vector,
                            )
                            if bf16_tile:
                                S.vsts(
                                    dequantized_ub[row, col],
                                    S.vshrs(T.reinterpret(widened, "uint32x64"), 16),
                                    dist="PK_B32",
                                )
                            else:
                                S.vsts(dequantized_ub[row, col], widened)

    @T.macro
    def clear_amax(amax_ub):
        with T.SimdVF():
            if bf16_amax:
                for sf_batch in T.unroll(amax_slots // 128, explicit=True):
                    S.vsts(
                        amax_ub[sf_batch * 128],
                        S.vdup(0.0, T.bfloat16),
                    )
            else:
                for sf_batch in T.unroll(num_sf_slots // 64, explicit=True):
                    S.vsts(
                        amax_ub[sf_batch * 64],
                        S.vdup(0.0, T.float32),
                    )

    @T.macro
    def reduce_groups(x_ub, amax_ub):
        with T.SimdVF():
            if group_size == 16 and not is_col_major_sf:
                amax_store_mask = S.pset(32, "PAT_VL8")
                if is_bf16_input:
                    abs_mask_u16 = S.vdup(0x7FFF, T.uint16)
                    zero_bf16 = S.vdup(0.0, T.bfloat16)
                for row in T.serial(block_m):
                    for tile in _lim(block_k // 128):
                        col = tile * 128
                        sf_offset = sf_slot(row, tile * 8)
                        if is_bf16_input:
                            values = S.vld(x_ub[row, col])
                            abs_u16 = S.vand(
                                T.reinterpret(values, "uint16x128"),
                                abs_mask_u16,
                            )
                            amax_u16 = S.vcgmax(abs_u16)
                            amax_low, _ = S.vintlv(
                                zero_bf16,
                                T.reinterpret(amax_u16, "bfloat16x128"),
                            )
                            amax = T.reinterpret(amax_low, "float32x64")
                        else:
                            values_even, values_odd = S.vld2(
                                x_ub[row, col],
                                dist="DINTLV_B32",
                            )
                            pair_amax = S.vmax(
                                S.vabs(values_even),
                                S.vabs(values_odd),
                            )
                            amax = S.vcgmax(pair_amax)
                        S.vsts(
                            amax_ub[sf_offset],
                            amax,
                            amax_store_mask,
                            dist="NORM_B32",
                            extent=8,
                        )
            elif group_size == 16:
                mask_0 = S.pset(32, "PAT_VL16")
                mask_01 = S.pset(32, "PAT_VL32")
                mask_all = S.pset(32, "PAT_ALL")
                mask_1 = S.pnot(mask_0, mask_01)
                mask_23 = S.pnot(mask_01, mask_all)
                mask_012 = S.vcmps(S.vci(0, T.int32), 48, op="lt")
                mask_2 = S.pand(mask_012, mask_23, mask_all)
                mask_3 = S.pnot(mask_2, mask_23)
                for row in T.serial(block_m):
                    for vector in _lim(block_k // 64):
                        values = S.vabs(load_as_fp32(x_ub, row, vector * 64))
                        S.vsts(
                            amax_ub[sf_slot(row, vector * 4)],
                            S.vcmax(values, mask_0),
                            dist="ONEPT_B32",
                        )
                        S.vsts(
                            amax_ub[sf_slot(row, vector * 4 + 1)],
                            S.vcmax(values, mask_1),
                            dist="ONEPT_B32",
                        )
                        S.vsts(
                            amax_ub[sf_slot(row, vector * 4 + 2)],
                            S.vcmax(values, mask_2),
                            dist="ONEPT_B32",
                        )
                        S.vsts(
                            amax_ub[sf_slot(row, vector * 4 + 3)],
                            S.vcmax(values, mask_3),
                            dist="ONEPT_B32",
                        )
            elif group_size == 32:
                mask_low = S.pset(32, "PAT_VL32")
                mask_high = S.vcmps(S.vci(0, T.int32), 31, op="gt")
                for row in T.serial(block_m):
                    for vector in _lim(block_k // 64):
                        values = S.vabs(load_as_fp32(x_ub, row, vector * 64))
                        S.vsts(
                            amax_ub[sf_slot(row, vector * 2)],
                            S.vcmax(values, mask_low),
                            dist="ONEPT_B32",
                        )
                        S.vsts(
                            amax_ub[sf_slot(row, vector * 2 + 1)],
                            S.vcmax(values, mask_high),
                            dist="ONEPT_B32",
                        )
            elif group_size == 64:
                for row in T.serial(block_m):
                    for vector in _lim(block_k // 64):
                        values = S.vabs(load_as_fp32(x_ub, row, vector * 64))
                        S.vsts(
                            amax_ub[sf_slot(row, vector)],
                            S.vcmax(values),
                            dist="ONEPT_B32",
                        )
            else:
                for row in T.serial(block_m):
                    for group in _lim(groups_per_tile):
                        col = group * 128
                        values0 = S.vabs(load_as_fp32(x_ub, row, col))
                        values1 = S.vabs(load_as_fp32(x_ub, row, col + 64))
                        S.vsts(
                            amax_ub[sf_slot(row, group)],
                            S.vcmax(S.vmax(values0, values1)),
                            dist="ONEPT_B32",
                        )

    @T.macro
    def reduce_groups_fast(x_ub, amax_ub):
        with T.SimdVF():
            lane = S.vci(0, T.int32)
            one = S.vdup(1, T.int32)
            base_index = S.vadd(
                S.vmuls(S.vshr(lane, one), block_m * 2),
                S.vand(lane, one),
            )
            store_mask = S.pset(16, "PAT_VL16") if bf16_amax else S.pset(32, "PAT_VL8")
            if reduce_reads_fp8:
                abs_mask_u16_pair = S.vdup(0x7F7F, T.uint16)
                low_byte_mask = S.vdup(0x00FF, T.uint16)
            elif reduce_reads_bf16:
                abs_mask_u16 = S.vdup(0x7FFF, T.uint16)
                if not bf16_amax:
                    zero_bf16 = S.vdup(0.0, T.bfloat16)
            rows_per_pass = 1 if block_k % 256 == 0 else 2
            for pass_row in T.serial(block_m // rows_per_pass):
                row = pass_row * rows_per_pass
                for chunk in _lim(max(block_k // 256, 1)):
                    col = chunk * 256
                    if reduce_reads_fp8:
                        magnitudes = S.vand(
                            T.reinterpret(S.vld(x_ub[row, col]), "uint16x128"),
                            abs_mask_u16_pair,
                        )
                        per_lane = S.vmax(
                            S.vshrs(magnitudes, 8),
                            S.vand(magnitudes, low_byte_mask),
                        )
                        winners = S.vcgmax(per_lane)
                        if bf16_amax:
                            group_max = S.vcvt(
                                T.reinterpret(winners, f"{input_dtype}x256"),
                                T.bfloat16,
                                part=0,
                            )
                        else:
                            stride4, _high = S.vintlv(winners, S.vdup(0, T.uint16))
                            group_max = S.vcvt(
                                T.reinterpret(stride4, f"{input_dtype}x256"),
                                T.float32,
                                part=0,
                            )
                    elif reduce_reads_bf16:
                        even, odd = S.vld2(x_ub[row, col], dist="DINTLV_B16")
                        pairs = S.vmax(
                            S.vand(T.reinterpret(even, "uint16x128"), abs_mask_u16),
                            S.vand(T.reinterpret(odd, "uint16x128"), abs_mask_u16),
                        )
                        if bf16_amax:
                            group_max = T.reinterpret(
                                S.vcgmax(pairs),
                                "bfloat16x128",
                            )
                        else:
                            amax_low, _amax_high = S.vintlv(
                                zero_bf16,
                                T.reinterpret(S.vcgmax(pairs), "bfloat16x128"),
                            )
                            group_max = T.reinterpret(amax_low, "float32x64")
                    else:
                        even_low, odd_low = S.vld2(x_ub[row, col], dist="DINTLV_B32")
                        if rows_per_pass == 2:
                            even_high, odd_high = S.vld2(
                                x_ub[row + 1, 0],
                                dist="DINTLV_B32",
                            )
                        else:
                            even_high, odd_high = S.vld2(
                                x_ub[row, col + 128],
                                dist="DINTLV_B32",
                            )
                        pairs_low = S.vmax(S.vabs(even_low), S.vabs(odd_low))
                        pairs_high = S.vmax(S.vabs(even_high), S.vabs(odd_high))
                        quads_even, quads_odd = S.vdintlv(pairs_low, pairs_high)
                        group_max = S.vcgmax(S.vmax(quads_even, quads_odd))
                    if amax_row_major:
                        S.vsts(
                            amax_ub[row * amax_row_pitch + chunk * 8],
                            group_max,
                            store_mask,
                            dist="NORM_B16" if bf16_amax else "NORM_B32",
                            extent=16 if bf16_amax else 8,
                        )
                    else:
                        S.vscatter(
                            group_max,
                            amax_ub[0],
                            T.reinterpret(
                                S.vadds(
                                    base_index,
                                    chunk * 4 * block_m * 2 + row * 2,
                                ),
                                "uint32x64",
                            ),
                            store_mask,
                        )

    @T.macro
    def reduce_groups_packed_row_micro(x_ub, amax_ub):
        with T.SimdVF():
            if group_size == 16:
                lane_ids = S.vci(0, T.int32)
                full_mask = S.pset(32, "PAT_ALL")
                for row in T.serial(block_m):
                    for micro in T.serial(packed_row_micro_tiles):
                        for local_vector in T.unroll(
                            packed_row_micro_vectors,
                            explicit=True,
                        ):
                            vector = micro * packed_row_micro_vectors + local_vector
                            values = S.vabs(load_as_fp32(x_ub, row, vector * 64))
                            for group in T.unroll(4, explicit=True):
                                lower = group * 16
                                mask_ge = S.vcmps(lane_ids, lower, op="ge")
                                mask_lt = S.vcmps(lane_ids, lower + 16, op="lt")
                                mask = S.pand(mask_ge, mask_lt, full_mask)
                                S.vsts(
                                    amax_ub[sf_slot(row, vector * 4 + group)],
                                    S.vcmax(values, mask),
                                    dist="ONEPT_B32",
                                )
            elif group_size == 32:
                mask_low = S.pset(32, "PAT_VL32")
                mask_high = S.vcmps(S.vci(0, T.int32), 31, op="gt")
                for row in T.serial(block_m):
                    for micro in T.serial(packed_row_micro_tiles):
                        for local_vector in T.unroll(
                            packed_row_micro_vectors,
                            explicit=True,
                        ):
                            vector = micro * packed_row_micro_vectors + local_vector
                            values = S.vabs(load_as_fp32(x_ub, row, vector * 64))
                            S.vsts(
                                amax_ub[sf_slot(row, vector * 2)],
                                S.vcmax(values, mask_low),
                                dist="ONEPT_B32",
                            )
                            S.vsts(
                                amax_ub[sf_slot(row, vector * 2 + 1)],
                                S.vcmax(values, mask_high),
                                dist="ONEPT_B32",
                            )
            elif group_size == 64:
                for row in T.serial(block_m):
                    for micro in T.serial(packed_row_micro_tiles):
                        for local_vector in T.unroll(
                            packed_row_micro_vectors,
                            explicit=True,
                        ):
                            vector = micro * packed_row_micro_vectors + local_vector
                            values = S.vabs(load_as_fp32(x_ub, row, vector * 64))
                            S.vsts(
                                amax_ub[sf_slot(row, vector)],
                                S.vcmax(values),
                                dist="ONEPT_B32",
                            )
            else:
                groups_per_micro = packed_row_micro_k // group_size
                for row in T.serial(block_m):
                    for micro in T.serial(packed_row_micro_tiles):
                        for local_group in T.unroll(groups_per_micro, explicit=True):
                            group = micro * groups_per_micro + local_group
                            col = group * 128
                            values0 = S.vabs(load_as_fp32(x_ub, row, col))
                            values1 = S.vabs(load_as_fp32(x_ub, row, col + 64))
                            S.vsts(
                                amax_ub[sf_slot(row, group)],
                                S.vcmax(S.vmax(values0, values1)),
                                dist="ONEPT_B32",
                            )

    @T.macro
    def make_amax_gather_base():
        if bf16_amax:
            lane = S.vand(S.vci(0, T.int16), S.vdup(63, T.int16))
            one = S.vdup(1, T.int16)
        else:
            lane = S.vci(0, T.int32)
            one = S.vdup(1, T.int32)
        if amax_needs_gather and not is_packed_sf:
            return S.vmuls(lane, amax_row_pitch)
        return S.vadd(
            S.vmuls(S.vshr(lane, one), amax_row_pitch),
            S.vand(lane, one),
        )

    @T.macro
    def scale_batch(
        amax_ub,
        sf_linear_ub,
        input_sf_values_ub,
        sf_batch,
        amax_gather_base,
        input_scale_lanes,
    ):
        sf_offset = sf_batch * 64
        if amax_needs_gather:
            if is_packed_sf:
                pair = sf_offset // (block_m * 2)
                rem_base = sf_offset % (block_m * 2)
                gather_offset = rem_base // 2 * amax_row_pitch + pair * 2
            else:
                gather_offset = sf_offset % block_m * amax_row_pitch + sf_offset // block_m
            batch_index = S.vadds(amax_gather_base, gather_offset)
            if bf16_amax:
                amax_low, _amax_high = S.vintlv(
                    S.vdup(0.0, T.bfloat16),
                    S.vgather2(
                        amax_ub[0],
                        T.reinterpret(batch_index, "uint16x128"),
                    ),
                )
                raw_amax = T.reinterpret(amax_low, "float32x64")
            else:
                raw_amax = S.vgather2(
                    amax_ub[0],
                    T.reinterpret(batch_index, "uint32x64"),
                )
        else:
            raw_amax = S.vld(amax_ub[sf_offset])
        group_amax = (
            S.vmul(
                raw_amax,
                tile_input_scale(input_sf_values_ub, sf_batch, input_scale_lanes),
            )
            if fuse_input_scale
            else raw_amax
        )
        amax = S.vmaxs(
            group_amax,
            out_config.clamp_min_value,
        )
        if out_config.round_sf:
            scale_raw = S.vmuls(amax, 1.0 / quant_max)
            scale_bits = T.reinterpret(scale_raw, "uint32x64")
            scale_exp = S.vadds(
                S.vshrs(S.vsub(scale_bits, S.vdup(1, T.uint32)), 23),
                1,
            )
            if not sf_only:
                inv_exp = S.vsub(S.vdup(254, T.uint32), scale_exp)
                inv_raw = T.reinterpret(S.vshls(inv_exp, 23), "float32x64")
                scale_inv = (
                    S.vmul(
                        inv_raw,
                        tile_input_scale(input_sf_values_ub, sf_batch, input_scale_lanes),
                    )
                    if fuse_input_scale
                    else inv_raw
                )
            if is_packed_sf:
                S.vsts(sf_linear_ub[sf_offset], scale_exp, dist="PK4_B32")
            else:
                scale = T.reinterpret(S.vshls(scale_exp, 23), "float32x64")
                S.vsts(sf_linear_ub[sf_offset], scale)
        else:
            quant_max_vec = S.vdup(quant_max, T.float32)
            scale = S.vdiv(amax, quant_max_vec)
            if not sf_only:
                scale_inv = S.vdiv(quant_max_vec, amax)
            S.vsts(sf_linear_ub[sf_offset], scale)
        if not sf_only:
            return scale_inv

    @T.macro
    def store_inverse_bf16(sf_inv_bf16_ub, slot, scale_inv):
        doubled = S.vor(
            T.reinterpret(S.vcvt(scale_inv, T.bfloat16), "uint16x128"),
            T.reinterpret(
                S.vcvt(scale_inv, T.bfloat16, part=1),
                "uint16x128",
            ),
        )
        S.vsts(
            sf_inv_bf16_ub[sf_inv_bf16_slot(slot)],
            T.reinterpret(doubled, "bfloat16x128"),
        )

    @T.macro
    def compute_scales(
        amax_ub,
        sf_linear_ub,
        sf_inv_ub,
        sf_inv_bf16_ub,
        input_sf_values_ub,
    ):
        with T.SimdVF():
            amax_gather_base = make_amax_gather_base()
            if fuse_input_scale_per_lane:
                input_scale_lanes = make_input_scale_lanes(input_sf_values_ub)
            if reorder_inverse_rows:
                row_blocks = block_m // 32
                pairs = S.alloc_local((4,), T.float32)
                rows = S.alloc_local((4,), T.float32)
                for row_block in T.unroll(row_blocks, explicit=True):
                    for pair in T.unroll(4, explicit=True):
                        pairs[pair] = scale_batch(
                            amax_ub,
                            sf_linear_ub,
                            input_sf_values_ub,
                            pair * row_blocks + row_block,
                            amax_gather_base,
                            input_scale_lanes if fuse_input_scale_per_lane else None,
                        )
                    even_low, odd_low = S.vdintlv(pairs[0], pairs[1])
                    even_high, odd_high = S.vdintlv(pairs[2], pairs[3])
                    even_a, even_b = S.vintlv(even_low, even_high)
                    even_rows_low, even_rows_high = S.vintlv(even_a, even_b)
                    odd_a, odd_b = S.vintlv(odd_low, odd_high)
                    odd_rows_low, odd_rows_high = S.vintlv(odd_a, odd_b)
                    rows[0], rows[1] = S.vintlv(even_rows_low, odd_rows_low)
                    rows[2], rows[3] = S.vintlv(even_rows_high, odd_rows_high)
                    for quarter in T.unroll(4, explicit=True):
                        store_inverse_bf16(
                            sf_inv_bf16_ub,
                            row_block * 256 + quarter * 64,
                            rows[quarter],
                        )
            else:
                for sf_batch in T.unroll(num_sf_slots // 64, explicit=True):
                    scale_inv = scale_batch(
                        amax_ub,
                        sf_linear_ub,
                        input_sf_values_ub,
                        sf_batch,
                        amax_gather_base,
                        input_scale_lanes if fuse_input_scale_per_lane else None,
                    )
                    if bf16_inverse:
                        store_inverse_bf16(
                            sf_inv_bf16_ub,
                            sf_batch * 64,
                            scale_inv,
                        )
                    elif not sf_only:
                        S.vsts(sf_inv_ub[sf_inv_slot(sf_batch * 64)], scale_inv)

    @T.macro
    def compute_inverse_scales(sf_linear_ub, sf_inv_ub):
        with T.SimdVF():
            for sf_batch in T.unroll(num_sf_slots // 64, explicit=True):
                sf_offset = sf_batch * 64
                if is_packed_sf:
                    scale_exp = T.reinterpret(
                        S.vld(sf_linear_ub[sf_offset], dist="UNPK4_B8"),
                        "uint32x64",
                    )
                    inv_exp = S.vsub(S.vdup(254, T.uint32), scale_exp)
                    scale_inv = T.reinterpret(S.vshls(inv_exp, 23), "float32x64")
                else:
                    scale = S.vld(sf_linear_ub[sf_offset])
                    scale_inv = S.vdiv(S.vdup(1.0, T.float32), scale)
                S.vsts(sf_inv_ub[sf_inv_slot(sf_offset)], scale_inv)

    @T.macro
    def load_col_major_inverse(sf_inv_ub, row, vector):
        if group_size == 16:
            lane_ids = S.vci(0, T.int32)
            inverses = S.alloc_local((4,), T.float32)
            for group in T.unroll(4, explicit=True):
                inverses[group] = S.vld(
                    sf_inv_ub[sf_inv_slot(sf_slot(row, vector * 4 + group))],
                    dist="BRC_B32",
                )
            inverse_low = S.vsel(
                inverses[0],
                inverses[1],
                S.vcmps(lane_ids, 16, op="lt"),
            )
            inverse_high = S.vsel(
                inverses[2],
                inverses[3],
                S.vcmps(lane_ids, 48, op="lt"),
            )
            return S.vsel(
                inverse_low,
                inverse_high,
                S.vcmps(lane_ids, 32, op="lt"),
            )
        if group_size == 32:
            mask_low = S.pset(32, "PAT_VL32")
            inverse_low = S.vld(
                sf_inv_ub[sf_inv_slot(sf_slot(row, vector * 2))],
                dist="BRC_B32",
            )
            inverse_high = S.vld(
                sf_inv_ub[sf_inv_slot(sf_slot(row, vector * 2 + 1))],
                dist="BRC_B32",
            )
            return S.vsel(
                inverse_low,
                inverse_high,
                mask_low,
            )
        if group_size == 64:
            return S.vld(
                sf_inv_ub[sf_inv_slot(sf_slot(row, vector))],
                dist="BRC_B32",
            )
        return S.vld(
            sf_inv_ub[sf_inv_slot(sf_slot(row, vector // 2))],
            dist="BRC_B32",
        )

    _qn = min(quantize_micro_vectors, block_k // 64)
    _qtiles = block_k // 64 // _qn

    @T.macro
    def _quantize_chunk(x_ub, sf_inv_ub, out_ub, token_base, hidden_base, row, _m):
        inverses = S.alloc_local((_qn,), T.float32)
        if is_col_major_sf:
            for vector in T.unroll(_qn, explicit=True):
                inverses[vector] = load_col_major_inverse(
                    sf_inv_ub,
                    row,
                    _m * _qn + vector,
                )
        elif group_size == 16:
            for pair in T.unroll(_qn // 2, explicit=True):
                expanded = S.vld(
                    sf_inv_ub[sf_inv_slot(row * groups_per_tile + (_m * (_qn // 2) + pair) * 8)],
                    dist="E2B_B32",
                )
                inverses[pair * 2], inverses[pair * 2 + 1] = S.vintlv(
                    expanded,
                    expanded,
                )
        elif group_size == 32:
            for batch in T.unroll(_qn // 4, explicit=True):
                expanded = S.vld(
                    sf_inv_ub[sf_inv_slot(row * groups_per_tile + (_m * (_qn // 4) + batch) * 8)],
                    dist="E2B_B32",
                )
                low, high = S.vintlv(expanded, expanded)
                inverses[batch * 4], inverses[batch * 4 + 1] = S.vintlv(
                    low,
                    low,
                )
                inverses[batch * 4 + 2], inverses[batch * 4 + 3] = S.vintlv(
                    high,
                    high,
                )
        elif group_size == 64:
            for vector in T.unroll(_qn, explicit=True):
                inverses[vector] = S.vld(
                    sf_inv_ub[sf_inv_slot(row * groups_per_tile + _m * _qn + vector)],
                    dist="BRC_B32",
                )
        else:
            for group in T.unroll(_qn // 2, explicit=True):
                inverse = S.vld(
                    sf_inv_ub[sf_inv_slot(row * groups_per_tile + _m * (_qn // 2) + group)],
                    dist="BRC_B32",
                )
                inverses[group * 2] = inverse
                inverses[group * 2 + 1] = inverse

        if is_fp4:
            scaled = S.alloc_local((2,), T.float32)
            for pair in T.unroll(_qn // 2, explicit=True):
                for half in T.unroll(2, explicit=True):
                    vector = pair * 2 + half
                    values = load_as_fp32(x_ub, row, (_m * _qn + vector) * 64)
                    scaled[half] = S.vmul(values, inverses[vector])
                    if stochastic_cast:
                        scaled[half] = _stochastic_round_vector(
                            scaled[half],
                            (token_base + row) * hidden + hidden_base + (_m * _qn + vector) * 64,
                            quant_max,
                            1,
                            -1,
                        )
                low, high = S.vdintlv(
                    T.reinterpret(scaled[0], "uint16x128"),
                    T.reinterpret(scaled[1], "uint16x128"),
                )
                scaled_bf16 = T.reinterpret(
                    S.vor(high, S.vmins(low, 1)),
                    "bfloat16x128",
                )
                quantized = S.vcvt(scaled_bf16, T.float4_e2m1fn)
                S.vsts(
                    out_ub[row, (_m * (_qn // 2) + pair) * 128],
                    quantized,
                    dist="PK4_B32",
                )
        else:
            for vector in T.unroll(_qn, explicit=True):
                values = load_as_fp32(x_ub, row, (_m * _qn + vector) * 64)
                scaled_values = S.vmul(values, inverses[vector])
                values_to_quantize = (
                    _stochastic_round_vector(
                        scaled_values,
                        (token_base + row) * hidden + hidden_base + (_m * _qn + vector) * 64,
                        quant_max,
                        3,
                        -9,
                    )
                    if stochastic_cast
                    else scaled_values
                )
                quantized = S.vcvt(
                    values_to_quantize,
                    T.float8_e4m3fn,
                )
                S.vsts(
                    out_ub[row, (_m * _qn + vector) * 64],
                    quantized,
                    dist="PK4_B32",
                )

    @T.macro
    def quantize(x_ub, sf_inv_ub, out_ub, token_base, hidden_base):
        with T.SimdVF():
            for row in T.serial(block_m):
                if _qtiles > 1:
                    for _micro in T.serial(_qtiles):
                        _quantize_chunk(x_ub, sf_inv_ub, out_ub, token_base, hidden_base, row, _micro)
                else:
                    _quantize_chunk(x_ub, sf_inv_ub, out_ub, token_base, hidden_base, row, 0)

    @T.macro
    def quantize_bf16(x_ub, sf_inv_bf16_ub, out_ub):
        with T.SimdVF():
            if row_major_inverse:
                for row in T.serial(block_m):
                    _mc = min(quantize_bf16_micro, block_k // 128)
                    for _micro in T.serial(block_k // 128 // _mc):
                        for _local in T.unroll(_mc, explicit=True):
                            chunk = _micro * _mc + _local
                            inverse = S.vld(
                                sf_inv_bf16_ub[sf_inv_bf16_slot(row * groups_per_tile + chunk * 4)],
                                dist="E2B_B16",
                            )
                            values = S.vld(x_ub[row, chunk * 128])
                            S.vsts(
                                out_ub[row, chunk * 128],
                                S.vcvt(S.vmul(values, inverse), T.float4_e2m1fn),
                                dist="PK4_B32",
                            )
            elif group_size >= 128:
                for row in T.serial(block_m):
                    _mc = min(quantize_bf16_micro, block_k // 128)
                    for _micro in T.serial(block_k // 128 // _mc):
                        for _local in T.unroll(_mc, explicit=True):
                            chunk = _micro * _mc + _local
                            inverse = T.reinterpret(
                                S.vld(
                                    sf_inv_bf16_ub[sf_inv_bf16_slot(sf_slot(row, chunk * 128 // group_size))],
                                    dist="BRC_B32",
                                ),
                                "bfloat16x128",
                            )
                            values = S.vld(x_ub[row, chunk * 128])
                            S.vsts(
                                out_ub[row, chunk * 128],
                                S.vcvt(S.vmul(values, inverse), T.float4_e2m1fn),
                                dist="PK4_B32",
                            )
            elif group_size == 64:
                lanes = S.vci(0, T.int16)
                low_half = S.vcmps(lanes, 64, op="lt")
                for row in T.serial(block_m):
                    _mc = min(quantize_bf16_micro, block_k // 128)
                    for _micro in T.serial(block_k // 128 // _mc):
                        for _local in T.unroll(_mc, explicit=True):
                            chunk = _micro * _mc + _local
                            group = chunk * 2
                            first = T.reinterpret(
                                S.vld(
                                    sf_inv_bf16_ub[sf_inv_bf16_slot(sf_slot(row, group))],
                                    dist="BRC_B32",
                                ),
                                "bfloat16x128",
                            )
                            second = T.reinterpret(
                                S.vld(
                                    sf_inv_bf16_ub[sf_inv_bf16_slot(sf_slot(row, group + 1))],
                                    dist="BRC_B32",
                                ),
                                "bfloat16x128",
                            )
                            inverse = S.vsel(first, second, low_half)
                            values = S.vld(x_ub[row, chunk * 128])
                            S.vsts(
                                out_ub[row, chunk * 128],
                                S.vcvt(S.vmul(values, inverse), T.float4_e2m1fn),
                                dist="PK4_B32",
                            )
            else:
                lanes = S.vci(0, T.int16)
                low_quarter = S.vcmps(lanes, 32, op="lt")
                low_half = S.vcmps(lanes, 64, op="lt")
                low_three_quarters = S.vcmps(lanes, 96, op="lt")
                for row in T.serial(block_m):
                    _mc = min(quantize_bf16_micro, block_k // 128)
                    for _micro in T.serial(block_k // 128 // _mc):
                        for _local in T.unroll(_mc, explicit=True):
                            chunk = _micro * _mc + _local
                            group = chunk * 4
                            first = T.reinterpret(
                                S.vld(
                                    sf_inv_bf16_ub[sf_inv_bf16_slot(sf_slot(row, group))],
                                    dist="BRC_B32",
                                ),
                                "bfloat16x128",
                            )
                            second = T.reinterpret(
                                S.vld(
                                    sf_inv_bf16_ub[sf_inv_bf16_slot(sf_slot(row, group + 1))],
                                    dist="BRC_B32",
                                ),
                                "bfloat16x128",
                            )
                            third = T.reinterpret(
                                S.vld(
                                    sf_inv_bf16_ub[sf_inv_bf16_slot(sf_slot(row, group + 2))],
                                    dist="BRC_B32",
                                ),
                                "bfloat16x128",
                            )
                            fourth = T.reinterpret(
                                S.vld(
                                    sf_inv_bf16_ub[sf_inv_bf16_slot(sf_slot(row, group + 3))],
                                    dist="BRC_B32",
                                ),
                                "bfloat16x128",
                            )
                            inverse = S.vsel(
                                S.vsel(first, second, low_quarter),
                                S.vsel(third, fourth, low_three_quarters),
                                low_half,
                            )
                            values = S.vld(x_ub[row, chunk * 128])
                            S.vsts(
                                out_ub[row, chunk * 128],
                                S.vcvt(S.vmul(values, inverse), T.float4_e2m1fn),
                                dist="PK4_B32",
                            )

    @T.macro
    def quantize_packed_row_micro(
        x_ub,
        sf_inv_ub,
        out_ub,
        token_base,
        hidden_base,
    ):
        with T.SimdVF():
            for row in T.serial(block_m):
                sf_base = row * groups_per_tile
                for micro in T.serial(packed_row_micro_tiles):
                    inverses = S.alloc_local((packed_row_micro_vectors,), T.float32)
                    vector_base = micro * packed_row_micro_vectors
                    if group_size == 16:
                        for pair in T.unroll(2, explicit=True):
                            expanded = S.vld(
                                sf_inv_ub[sf_inv_slot(sf_base + micro * 16 + pair * 8)],
                                dist="E2B_B32",
                            )
                            inverses[pair * 2], inverses[pair * 2 + 1] = S.vintlv(
                                expanded,
                                expanded,
                            )
                    elif group_size == 32:
                        expanded = S.vld(
                            sf_inv_ub[sf_inv_slot(sf_base + micro * 8)],
                            dist="E2B_B32",
                        )
                        low, high = S.vintlv(expanded, expanded)
                        inverses[0], inverses[1] = S.vintlv(low, low)
                        inverses[2], inverses[3] = S.vintlv(high, high)
                    elif group_size == 64:
                        for local_vector in T.unroll(
                            packed_row_micro_vectors,
                            explicit=True,
                        ):
                            inverses[local_vector] = S.vld(
                                sf_inv_ub[sf_inv_slot(sf_base + vector_base + local_vector)],
                                dist="BRC_B32",
                            )
                    else:
                        for local_group in T.unroll(2, explicit=True):
                            inverse = S.vld(
                                sf_inv_ub[sf_inv_slot(sf_base + micro * 2 + local_group)],
                                dist="BRC_B32",
                            )
                            inverses[local_group * 2] = inverse
                            inverses[local_group * 2 + 1] = inverse

                    if is_fp4:
                        scaled = S.alloc_local((2,), T.float32)
                        for local_pair in T.unroll(2, explicit=True):
                            for half in T.unroll(2, explicit=True):
                                local_vector = local_pair * 2 + half
                                vector = vector_base + local_vector
                                values = load_as_fp32(x_ub, row, vector * 64)
                                scaled[half] = S.vmul(values, inverses[local_vector])
                                if stochastic_cast:
                                    scaled[half] = _stochastic_round_vector(
                                        scaled[half],
                                        (token_base + row) * hidden + hidden_base + vector * 64,
                                        quant_max,
                                        1,
                                        -1,
                                    )
                            low, high = S.vdintlv(
                                T.reinterpret(scaled[0], "uint16x128"),
                                T.reinterpret(scaled[1], "uint16x128"),
                            )
                            scaled_bf16 = T.reinterpret(
                                S.vor(high, S.vmins(low, 1)),
                                "bfloat16x128",
                            )
                            quantized = S.vcvt(scaled_bf16, T.float4_e2m1fn)
                            S.vsts(
                                out_ub[row, micro * packed_row_micro_k + local_pair * 128],
                                quantized,
                                dist="PK4_B32",
                            )
                    else:
                        for local_vector in T.unroll(
                            packed_row_micro_vectors,
                            explicit=True,
                        ):
                            vector = vector_base + local_vector
                            values = load_as_fp32(x_ub, row, vector * 64)
                            scaled_values = S.vmul(values, inverses[local_vector])
                            values_to_quantize = (
                                _stochastic_round_vector(
                                    scaled_values,
                                    (token_base + row) * hidden + hidden_base + vector * 64,
                                    quant_max,
                                    3,
                                    -9,
                                )
                                if stochastic_cast
                                else scaled_values
                            )
                            quantized = S.vcvt(
                                values_to_quantize,
                                T.float8_e4m3fn,
                            )
                            S.vsts(
                                out_ub[row, vector * 64],
                                quantized,
                                dist="PK4_B32",
                            )

    @T.prim_func
    def per_token_cast_grouped(
        x: T.StridedTensor[
            (num_tokens, hidden),
            (token_stride, 1),
            input_dtype,
        ],
        x_sf: T.Tensor[x_sf_shape, in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[
            sf_shape,
            (out_sf_stride, 1),
            out_config.sf_dtype,
        ],
    ):
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), input_dtype)
            if has_input_sf:
                x_ub_uint8 = T.Tensor(
                    (block_m, block_k // 2 if is_fp4_input else block_k),
                    T.uint8,
                    x_ub.data,
                )
                input_sf_raw_ub = T.alloc_shared(input_sf_raw_shape, input_sf_raw_dtype)
                input_sf_raw_linear_ub = T.Tensor(
                    (input_sf_raw_slots,),
                    input_sf_raw_dtype,
                    input_sf_raw_ub.data,
                )
                input_sf_values_ub = T.alloc_shared(
                    (input_sf_value_slots,),
                    T.float32,
                )
                input_sf_bf16_ub = T.Tensor(
                    (input_sf_value_slots * 2,),
                    T.bfloat16,
                    input_sf_values_ub.data,
                )
                if not no_dequant_tile:
                    dequantized_ub = T.alloc_shared((block_m, block_k), staging_dtype)
            amax_ub = T.alloc_shared((amax_slots,), amax_dtype)
            sf_workspace_ub = T.alloc_shared(
                (sf_workspace_words,),
                T.float32,
            )
            sf_storage_ub = T.SharedBuffer(
                (num_sf_slots,),
                sf_ub_dtype,
                data=sf_workspace_ub.data,
                elem_offset=0,
            )
            sf_ub = T.Tensor(sf_ub_shape, sf_ub_dtype, sf_storage_ub.data)
            sf_linear_ub = T.Tensor(
                (num_sf_slots,),
                sf_ub_dtype,
                sf_storage_ub.data,
            )
            sf_inv_ub = sf_workspace_ub
            sf_inv_bf16_ub = T.Tensor(
                (sf_workspace_words * 2,),
                T.bfloat16,
                sf_workspace_ub.data,
            )
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)

            buffer_versions = {
                x_ub: num_stages,
                sf_workspace_ub: num_stages,
                out_ub: num_stages,
            }
            if has_input_sf:
                buffer_versions[input_sf_raw_ub] = num_stages
            T.annotate_buffer_versions(buffer_versions)

            for token_tile, hidden_tile in T.Persistent(
                [T.ceildiv(num_tokens, block_m), num_hidden_tiles],
                num_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
                annotations=pipeline_offset_annotations,
            ):
                token_base = token_tile * block_m
                hidden_base = hidden_tile * block_k
                sf_base = hidden_tile * groups_per_tile
                valid_rows = T.min(block_m, num_tokens - token_base)
                valid_cols = T.min(block_k, hidden - hidden_base)
                valid_groups = T.ceildiv(valid_cols, group_size)

                with _stage(0, manual_pipeline_stages):
                    if hidden % block_k != 0 and hidden_tile == num_hidden_tiles - 1:
                        if has_input_sf:
                            clear_quantized_input(x_ub_uint8)
                        else:
                            clear_raw_input(x_ub)
                    T.copy(
                        x[
                            token_base : token_base + valid_rows,
                            hidden_base : hidden_base + valid_cols,
                        ],
                        x_ub[:valid_rows, :valid_cols],
                    )
                    if has_input_sf:
                        input_sf_row = token_base // input_block_m
                        input_sf_base = hidden_base // input_group_size
                        valid_input_groups = T.ceildiv(valid_cols, input_group_size)
                        if hidden % block_k != 0:
                            clear_input_scales(input_sf_raw_linear_ub)
                        if input_sf_col_major:
                            if in_config.use_packed_ue8m0:
                                valid_input_group_pairs = T.ceildiv(valid_input_groups, 2)
                                T.copy(
                                    x_sf[
                                        input_sf_base // 2 : input_sf_base // 2 + valid_input_group_pairs,
                                        input_sf_row * 2 : input_sf_row * 2 + 2,
                                    ],
                                    input_sf_raw_ub[:valid_input_group_pairs, :2],
                                )
                            else:
                                T.copy(
                                    x_sf[
                                        input_sf_base : input_sf_base + valid_input_groups,
                                        input_sf_row : input_sf_row + 1,
                                    ],
                                    input_sf_raw_ub[:valid_input_groups, :1],
                                )
                        else:
                            T.copy(
                                x_sf[
                                    input_sf_row : input_sf_row + 1,
                                    input_sf_base : input_sf_base + valid_input_groups,
                                ],
                                input_sf_raw_ub[:1, :valid_input_groups],
                            )
                    if not manual_pipeline_stages and has_input_sf:
                        decode_input_scales(
                            input_sf_raw_linear_ub,
                            input_sf_values_ub,
                        )
                        if not no_dequant_tile:
                            dequantize_input(
                                x_ub,
                                input_sf_values_ub,
                                input_sf_bf16_ub if input_scale_bf16 else None,
                                dequantized_ub,
                                amax_ub if fuse_amax_into_dequant else None,
                            )
                    if cast_only:
                        if is_col_major_sf:
                            if is_packed_sf:
                                valid_group_pairs = T.ceildiv(valid_groups, 2)
                                T.copy(
                                    out_sf[
                                        sf_base // 2 : sf_base // 2 + valid_group_pairs,
                                        token_base * 2 : token_base * 2 + valid_rows * 2,
                                    ],
                                    sf_ub[:valid_group_pairs, : valid_rows * 2],
                                )
                            else:
                                T.copy(
                                    out_sf[
                                        sf_base : sf_base + valid_groups,
                                        token_base : token_base + valid_rows,
                                    ],
                                    sf_ub[:valid_groups, :valid_rows],
                                )
                        else:
                            T.copy(
                                out_sf[
                                    token_base : token_base + valid_rows,
                                    sf_base : sf_base + valid_groups,
                                ],
                                sf_ub[:valid_rows, :valid_groups],
                            )

                with _stage(1, manual_pipeline_stages):
                    if manual_pipeline_stages and has_input_sf:
                        decode_input_scales(
                            input_sf_raw_linear_ub,
                            input_sf_values_ub,
                        )
                        if not no_dequant_tile:
                            dequantize_input(
                                x_ub,
                                input_sf_values_ub,
                                input_sf_bf16_ub if input_scale_bf16 else None,
                                dequantized_ub,
                                amax_ub if fuse_amax_into_dequant else None,
                            )
                    if cast_only:
                        compute_inverse_scales(sf_linear_ub, sf_inv_ub)
                    else:
                        if not fuse_amax_into_dequant:
                            clear_amax(amax_ub)
                            if fast_reduce:
                                reduce_groups_fast(
                                    dequantized_ub if has_input_sf and not reduce_reads_fp8 else x_ub,
                                    amax_ub,
                                )
                            elif has_input_sf:
                                reduce_groups(dequantized_ub, amax_ub)
                            elif use_packed_row_micro_loop:
                                reduce_groups_packed_row_micro(x_ub, amax_ub)
                            else:
                                reduce_groups(x_ub, amax_ub)
                        compute_scales(
                            amax_ub,
                            sf_linear_ub,
                            sf_inv_ub,
                            sf_inv_bf16_ub,
                            input_sf_values_ub if has_input_sf else amax_ub,
                        )
                    if not manual_pipeline_stages and not cast_only:
                        if is_col_major_sf:
                            if is_packed_sf:
                                valid_group_pairs = T.ceildiv(valid_groups, 2)
                                T.copy(
                                    sf_ub[:valid_group_pairs, : valid_rows * 2],
                                    out_sf[
                                        sf_base // 2 : sf_base // 2 + valid_group_pairs,
                                        token_base * 2 : token_base * 2 + valid_rows * 2,
                                    ],
                                )
                            else:
                                T.copy(
                                    sf_ub[:valid_groups, :valid_rows],
                                    out_sf[
                                        sf_base : sf_base + valid_groups,
                                        token_base : token_base + valid_rows,
                                    ],
                                )
                        else:
                            T.copy(
                                sf_ub[:valid_rows, :valid_groups],
                                out_sf[
                                    token_base : token_base + valid_rows,
                                    sf_base : sf_base + valid_groups,
                                ],
                            )
                    if not sf_only:
                        if has_input_sf and not no_dequant_tile:
                            quantize_input_ub = dequantized_ub
                        else:
                            quantize_input_ub = x_ub
                        if bf16_inverse:
                            quantize_bf16(
                                quantize_input_ub,
                                sf_inv_bf16_ub,
                                out_ub,
                            )
                        elif use_packed_row_micro_loop:
                            quantize_packed_row_micro(
                                quantize_input_ub,
                                sf_inv_ub,
                                out_ub,
                                token_base,
                                hidden_base,
                            )
                        else:
                            quantize(
                                quantize_input_ub,
                                sf_inv_ub,
                                out_ub,
                                token_base,
                                hidden_base,
                            )

                with _stage(2, manual_pipeline_stages):
                    if manual_pipeline_stages and not cast_only:
                        if is_col_major_sf:
                            if is_packed_sf:
                                valid_group_pairs = T.ceildiv(valid_groups, 2)
                                T.copy(
                                    sf_ub[:valid_group_pairs, : valid_rows * 2],
                                    out_sf[
                                        sf_base // 2 : sf_base // 2 + valid_group_pairs,
                                        token_base * 2 : token_base * 2 + valid_rows * 2,
                                    ],
                                )
                            else:
                                T.copy(
                                    sf_ub[:valid_groups, :valid_rows],
                                    out_sf[
                                        sf_base : sf_base + valid_groups,
                                        token_base : token_base + valid_rows,
                                    ],
                                )
                        else:
                            T.copy(
                                sf_ub[:valid_rows, :valid_groups],
                                out_sf[
                                    token_base : token_base + valid_rows,
                                    sf_base : sf_base + valid_groups,
                                ],
                            )
                    if not sf_only:
                        T.copy(
                            out_ub[:valid_rows, :valid_cols],
                            out[
                                token_base : token_base + valid_rows,
                                hidden_base : hidden_base + valid_cols,
                            ],
                        )

    return per_token_cast_grouped


@tilelang.jit
def _get_full_row_cast_kernel(
    hidden: int,
    token_stride: int,
    input_dtype: T.dtype,
    out_config: CastOutputConfig,
    sf_only: bool,
    cast_only: bool,
    stochastic_cast: bool,
):
    assert out_config.sf_block == (1, hidden)
    assert hidden % 64 == 0
    assert input_dtype in (T.float32, T.bfloat16)

    num_cores = get_num_vec_cores()
    block_k = 1024
    num_hidden_tiles = T.ceildiv(hidden, block_k)
    is_bf16_input = input_dtype == T.bfloat16
    is_fp4 = out_config.dtype == T.float4_e2m1fn
    is_packed_sf = out_config.use_packed_ue8m0
    is_col_major_sf = out_config.use_tma_aligned_col_major_sf
    sf_ub_dtype = T.uint8 if is_packed_sf else T.float32
    sf_cols = 2 if is_packed_sf else 1
    block_m = 16 if is_packed_sf else 8
    sf_ub_shape = (1, block_m * sf_cols) if is_col_major_sf else (block_m, sf_cols)
    sf_storage_slots = max(64, block_m * sf_cols)
    quant_max = 6.0 if is_fp4 else 448.0
    num_tokens = T.dynamic("num_tokens")
    out_sf_stride = T.dynamic("out_sf_stride")
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)

    @T.macro
    def load_as_fp32(x_ub, row, col):
        if is_bf16_input:
            return S.vcvt(S.vld(x_ub[row, col], dist="UNPK_B16"), T.float32, part=0)
        return S.vld(x_ub[row, col])

    @T.macro
    def clear_input(x_ub):
        with T.SimdVF():
            zero = S.vdup(0.0, input_dtype)
            for row in T.serial(block_m):
                for vector in T.unroll(
                    block_k // (128 if is_bf16_input else 64),
                    explicit=True,
                ):
                    S.vsts(
                        x_ub[row, vector * (128 if is_bf16_input else 64)],
                        zero,
                    )

    @T.macro
    def clear_amax(amax_ub):
        with T.SimdVF():
            for row in T.serial(block_m):
                S.vsts(amax_ub[row, 0], S.vdup(0.0, T.float32))

    @T.macro
    def update_amax(x_ub, amax_ub):
        with T.SimdVF():
            for row in T.serial(block_m):
                for chunk in T.unroll(block_k // 256, explicit=True):
                    col = chunk * 256
                    values0 = S.vabs(load_as_fp32(x_ub, row, col))
                    values1 = S.vabs(load_as_fp32(x_ub, row, col + 64))
                    values2 = S.vabs(load_as_fp32(x_ub, row, col + 128))
                    values3 = S.vabs(load_as_fp32(x_ub, row, col + 192))
                    tile_amax = S.vmax(
                        S.vmax(values0, values1),
                        S.vmax(values2, values3),
                    )
                    S.vsts(
                        amax_ub[row, 0],
                        S.vmax(S.vld(amax_ub[row, 0]), tile_amax),
                    )

    @T.macro
    def finish_scales(amax_ub, sf_storage_ub, sf_inv_ub):
        with T.SimdVF():
            for row in T.serial(block_m):
                amax = S.vmaxs(
                    S.vdupv(S.vcmax(S.vld(amax_ub[row, 0]))),
                    out_config.clamp_min_value,
                )
                if out_config.round_sf:
                    scale_raw = S.vmuls(amax, 1.0 / quant_max)
                    scale_bits = T.reinterpret(scale_raw, "uint32x64")
                    scale_exp = S.vadds(
                        S.vshrs(S.vsub(scale_bits, S.vdup(1, T.uint32)), 23),
                        1,
                    )
                    inv_exp = S.vsub(S.vdup(254, T.uint32), scale_exp)
                    scale_inv = T.reinterpret(S.vshls(inv_exp, 23), "float32x64")
                    if is_packed_sf:
                        S.vsts(
                            sf_storage_ub[row * 2],
                            T.reinterpret(scale_exp, "uint16x128"),
                            dist="ONEPT_B16",
                        )
                    else:
                        scale = T.reinterpret(S.vshls(scale_exp, 23), "float32x64")
                        S.vsts(
                            sf_storage_ub[row],
                            scale,
                            dist="ONEPT_B32",
                        )
                else:
                    quant_max_vec = S.vdup(quant_max, T.float32)
                    scale = S.vdiv(amax, quant_max_vec)
                    scale_inv = S.vdiv(quant_max_vec, amax)
                    S.vsts(
                        sf_storage_ub[row],
                        scale,
                        dist="ONEPT_B32",
                    )
                S.vsts(
                    sf_inv_ub[row, 0],
                    scale_inv,
                )

    @T.macro
    def compute_inverse_scales(sf_storage_ub, sf_inv_ub):
        with T.SimdVF():
            lane_ids = S.vci(0, T.int32)
            if is_packed_sf:
                scale_values = T.reinterpret(
                    S.vld(sf_storage_ub[0], dist="UNPK4_B8"),
                    "uint32x64",
                )
            else:
                scale_values = S.vld(sf_storage_ub[0])
            for row in T.serial(block_m):
                scale_lane = row * 2 if is_packed_sf else row
                lane_mask = S.vcmps(lane_ids, scale_lane, op="eq")
                if is_packed_sf:
                    scale_exp = S.vdupv(
                        S.vcmax(scale_values, lane_mask),
                    )
                    inv_exp = S.vsub(S.vdup(254, T.uint32), scale_exp)
                    inverse = T.reinterpret(S.vshls(inv_exp, 23), "float32x64")
                else:
                    scale = S.vdupv(
                        S.vcmax(scale_values, lane_mask),
                    )
                    inverse = S.vdiv(S.vdup(1.0, T.float32), scale)
                S.vsts(
                    sf_inv_ub[row, 0],
                    inverse,
                )

    @T.macro
    def quantize(x_ub, sf_inv_ub, out_ub, token_base, hidden_base):
        with T.SimdVF():
            scaled = S.alloc_local((2,), T.float32)
            for row in T.serial(block_m):
                inverse = S.vld(sf_inv_ub[row, 0])
                if is_fp4:
                    for pair in T.unroll(block_k // 128, explicit=True):
                        for half in T.unroll(2, explicit=True):
                            col = pair * 128 + half * 64
                            scaled[half] = S.vmul(
                                load_as_fp32(x_ub, row, col),
                                inverse,
                            )
                            if stochastic_cast:
                                scaled[half] = _stochastic_round_vector(
                                    scaled[half],
                                    (token_base + row) * hidden + hidden_base + col,
                                    quant_max,
                                    1,
                                    -1,
                                )
                        low, high = S.vdintlv(
                            T.reinterpret(scaled[0], "uint16x128"),
                            T.reinterpret(scaled[1], "uint16x128"),
                        )
                        scaled_bf16 = T.reinterpret(
                            S.vor(high, S.vmins(low, 1)),
                            "bfloat16x128",
                        )
                        quantized = S.vcvt(scaled_bf16, T.float4_e2m1fn)
                        S.vsts(
                            out_ub[row, pair * 128],
                            quantized,
                            dist="PK4_B32",
                        )
                else:
                    for vector in T.unroll(block_k // 64, explicit=True):
                        col = vector * 64
                        values = load_as_fp32(x_ub, row, col)
                        scaled_values = S.vmul(values, inverse)
                        values_to_quantize = (
                            _stochastic_round_vector(
                                scaled_values,
                                (token_base + row) * hidden + hidden_base + col,
                                quant_max,
                                3,
                                -9,
                            )
                            if stochastic_cast
                            else scaled_values
                        )
                        quantized = S.vcvt(
                            values_to_quantize,
                            T.float8_e4m3fn,
                        )
                        S.vsts(
                            out_ub[row, col],
                            quantized,
                            dist="PK4_B32",
                        )

    @T.prim_func
    def per_token_cast_full_row(
        x: T.StridedTensor[
            (num_tokens, hidden),
            (token_stride, 1),
            input_dtype,
        ],
        x_sf: T.Tensor[(1, 1), T.float32],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[
            sf_shape,
            (out_sf_stride, 1),
            out_config.sf_dtype,
        ],
    ):
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), input_dtype)
            amax_ub = T.alloc_shared((block_m, 64), T.float32)
            sf_storage_ub = T.alloc_shared((sf_storage_slots,), sf_ub_dtype)
            sf_ub = T.Tensor(sf_ub_shape, sf_ub_dtype, sf_storage_ub.data)
            sf_inv_ub = T.alloc_shared((block_m, 64), T.float32)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)

            for token_tile in T.Persistent(
                [T.ceildiv(num_tokens, block_m)],
                num_cores,
                core_id,
                group_size=1,
                num_stages=1,
            ):
                token_base = token_tile * block_m
                valid_rows = T.min(block_m, num_tokens - token_base)
                if cast_only:
                    if is_col_major_sf:
                        T.copy(
                            out_sf[
                                0:1,
                                token_base * sf_cols : token_base * sf_cols + valid_rows * sf_cols,
                            ],
                            sf_ub[:, : valid_rows * sf_cols],
                        )
                    else:
                        T.copy(
                            out_sf[
                                token_base : token_base + valid_rows,
                                :sf_cols,
                            ],
                            sf_ub[:valid_rows, :sf_cols],
                        )
                    compute_inverse_scales(sf_storage_ub, sf_inv_ub)
                else:
                    clear_amax(amax_ub)
                    for hidden_tile in T.serial(num_hidden_tiles):
                        hidden_base = hidden_tile * block_k
                        valid_cols = T.min(block_k, hidden - hidden_base)
                        if hidden % block_k != 0 and hidden_tile == num_hidden_tiles - 1:
                            clear_input(x_ub)
                        T.copy(
                            x[
                                token_base : token_base + valid_rows,
                                hidden_base : hidden_base + valid_cols,
                            ],
                            x_ub[:valid_rows, :valid_cols],
                        )
                        update_amax(x_ub, amax_ub)

                    finish_scales(amax_ub, sf_storage_ub, sf_inv_ub)
                    if is_col_major_sf:
                        T.copy(
                            sf_ub[:, : valid_rows * sf_cols],
                            out_sf[
                                0:1,
                                token_base * sf_cols : token_base * sf_cols + valid_rows * sf_cols,
                            ],
                        )
                    else:
                        T.copy(
                            sf_ub[:valid_rows, :sf_cols],
                            out_sf[
                                token_base : token_base + valid_rows,
                                :sf_cols,
                            ],
                        )

                if not sf_only:
                    for hidden_tile in T.serial(num_hidden_tiles):
                        hidden_base = hidden_tile * block_k
                        valid_cols = T.min(block_k, hidden - hidden_base)
                        if hidden % block_k != 0 and hidden_tile == num_hidden_tiles - 1:
                            clear_input(x_ub)
                        T.copy(
                            x[
                                token_base : token_base + valid_rows,
                                hidden_base : hidden_base + valid_cols,
                            ],
                            x_ub[:valid_rows, :valid_cols],
                        )
                        quantize(
                            x_ub,
                            sf_inv_ub,
                            out_ub,
                            token_base,
                            hidden_base,
                        )
                        T.copy(
                            out_ub[:valid_rows, :valid_cols],
                            out[
                                token_base : token_base + valid_rows,
                                hidden_base : hidden_base + valid_cols,
                            ],
                        )

    return per_token_cast_full_row


PREQUANT_ROW_MAJOR_ROWS = 64
PREQUANT_ROW_MAJOR_BLOCK_K = 512


def prequant_row_major_path_ok(hidden, in_config, out_config, sf_only, cast_only, stochastic_cast, small_batch) -> bool:
    if not in_config.with_sf:
        return False
    input_block_m, input_group_size = in_config.sf_block
    return (
        in_config.use_packed_ue8m0 == out_config.use_packed_ue8m0
        and in_config.use_tma_aligned_col_major_sf == out_config.use_tma_aligned_col_major_sf
        and out_config.round_sf
        and not stochastic_cast
        and not sf_only
        and not cast_only
        and out_config.dtype == T.float8_e4m3fn
        and out_config.sf_block[1] == 32
        and input_group_size == 32
        and hidden % PREQUANT_ROW_MAJOR_BLOCK_K == 0
        and input_block_m < PREQUANT_ROW_MAJOR_ROWS
        and PREQUANT_ROW_MAJOR_ROWS % input_block_m == 0
        and not small_batch
    )


@tilelang.jit
def _get_prequant_row_major_cast_kernel(
    hidden: int,
    token_stride: int,
    in_config: CastInputConfig,
    out_config: CastOutputConfig,
):
    input_dtype = in_config.dtype
    is_fp4_input = input_dtype == T.float4_e2m1fn
    is_fp4 = out_config.dtype == T.float4_e2m1fn
    is_col_major_sf = out_config.use_tma_aligned_col_major_sf
    is_packed_sf = out_config.use_packed_ue8m0
    input_block_m, input_group_size = in_config.sf_block
    group_size = out_config.sf_block[1]
    block_m = PREQUANT_ROW_MAJOR_ROWS
    block_k = PREQUANT_ROW_MAJOR_BLOCK_K
    num_stages = 2
    num_cores = get_num_vec_cores()
    num_hidden_tiles = hidden // block_k
    chunks_per_row = block_k // 256
    groups_per_tile = block_k // group_size
    groups_per_chunk = 256 // group_size
    subs_per_row = block_k // 32
    input_rows = block_m // input_block_m if input_block_m <= block_m else 1
    input_groups_per_tile = block_k // input_group_size
    quant_max = 6.0 if is_fp4 else 448.0

    pad = 64
    input_sf_pitch = 64
    sf_slots = block_m * groups_per_tile
    factor_offset = sf_slots + pad
    g128_stage_offset = factor_offset + block_m * subs_per_row + pad
    g128_stage_row_pitch = chunks_per_row * 8
    workspace_slots = g128_stage_offset + (block_m * g128_stage_row_pitch + pad if group_size == 128 else 0)

    num_tokens = T.dynamic("num_tokens")
    out_sf_stride = T.dynamic("out_sf_stride")
    x_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    sf_shape = get_sf_shape((num_tokens, hidden), out_config)

    def sf_store_slot(row, chunk):
        if group_size == 128:
            return g128_stage_offset + row * g128_stage_row_pitch + chunk * 8
        return row * groups_per_tile + chunk * groups_per_chunk

    def input_row_of(row):
        return row // input_block_m if input_block_m <= block_m else 0

    @T.macro
    def expand_input_scales(input_sf_linear_ub, input_sf32_ub):
        with T.SimdVF():
            for input_row in T.serial(input_rows):
                scales = S.vld(input_sf_linear_ub[input_row * input_sf_pitch])
                doubled, _doubled_high = S.vintlv(scales, scales)
                quadrupled, _quadrupled_high = S.vintlv(doubled, doubled)
                S.vsts(input_sf32_ub[input_row * input_sf_pitch], quadrupled)

    @T.macro
    def reduce_and_scale(x_ub, input_sf32_linear_ub, workspace_ub, sf_bytes_ub):
        with T.SimdVF():
            one = S.vdup(1, T.uint32)
            exp_limit = S.vdup(254, T.uint32)
            sf_store_mask = S.pset(32, "PAT_VL8" if groups_per_chunk == 8 else "PAT_VL2")
            factor_store_mask = S.pset(32, "PAT_VL8")
            if group_size == 128:
                zero_f32 = S.vdup(0.0, T.float32)
            if is_fp4_input:
                abs_mask = S.vdup(0x7FFF, T.uint16)
                zero_bf16 = S.vdup(0.0, T.bfloat16)
            else:
                abs_mask_pair = S.vdup(0x7F7F, T.uint16)
                low_byte = S.vdup(0x00FF, T.uint16)
                zero_u16 = S.vdup(0, T.uint16)
            for row in T.serial(block_m):
                for chunk in T.unroll(chunks_per_row, explicit=True):
                    col = chunk * 256
                    if is_fp4_input:
                        first = T.reinterpret(
                            S.vcvt(S.vld(x_ub[row, col], dist="UNPK4_B8"), T.bfloat16),
                            "uint16x128",
                        )
                        second = T.reinterpret(
                            S.vcvt(S.vld(x_ub[row, col + 128], dist="UNPK4_B8"), T.bfloat16),
                            "uint16x128",
                        )
                        even, odd = S.vdintlv(first, second)
                        pairs = S.vmax(S.vand(even, abs_mask), S.vand(odd, abs_mask))
                        winners, _winners_high = S.vintlv(
                            zero_bf16,
                            T.reinterpret(S.vcgmax(pairs), "bfloat16x128"),
                        )
                        code_max = T.reinterpret(winners, "float32x64")
                    else:
                        magnitudes = S.vand(
                            T.reinterpret(S.vld(x_ub[row, col]), "uint16x128"),
                            abs_mask_pair,
                        )
                        per_lane = S.vmax(S.vshrs(magnitudes, 8), S.vand(magnitudes, low_byte))
                        stride4, _stride4_high = S.vintlv(S.vcgmax(per_lane), zero_u16)
                        code_max = S.vcvt(
                            T.reinterpret(stride4, f"{input_dtype}x256"),
                            T.float32,
                            part=0,
                        )
                    input_scales = S.vld(input_sf32_linear_ub[input_row_of(row) * input_sf_pitch + chunk * 8])
                    scaled_max = S.vmul(code_max, input_scales)
                    if group_size == 128:
                        spread, _spread_high = S.vintlv(scaled_max, zero_f32)
                        amax = S.vmaxs(S.vcgmax(spread), out_config.clamp_min_value)
                    else:
                        amax = S.vmaxs(scaled_max, out_config.clamp_min_value)
                    scale_bits = T.reinterpret(S.vmuls(amax, 1.0 / quant_max), "uint32x64")
                    scale_exp = S.vadds(S.vshrs(S.vsub(scale_bits, one), 23), 1)
                    inverse = T.reinterpret(
                        S.vshls(S.vsub(exp_limit, scale_exp), 23),
                        "float32x64",
                    )
                    if is_packed_sf:
                        S.vsts(
                            workspace_ub[sf_store_slot(row, chunk)],
                            T.reinterpret(scale_exp, "float32x64"),
                            sf_store_mask,
                            dist="NORM_B32",
                            extent=groups_per_chunk,
                        )
                    else:
                        S.vsts(
                            workspace_ub[sf_store_slot(row, chunk)],
                            T.reinterpret(S.vshls(scale_exp, 23), "float32x64"),
                            sf_store_mask,
                            dist="NORM_B32",
                            extent=groups_per_chunk,
                        )
                    if group_size == 128:
                        inverse_pairs, _inverse_pairs_high = S.vintlv(inverse, inverse)
                        inverse_subs, _inverse_subs_high = S.vintlv(inverse_pairs, inverse_pairs)
                        factors = S.vmul(input_scales, inverse_subs)
                    else:
                        factors = S.vmul(input_scales, inverse)
                    S.vsts(
                        workspace_ub[factor_offset + row * subs_per_row + chunk * 8],
                        factors,
                        factor_store_mask,
                        dist="NORM_B32",
                        extent=8,
                    )

    @T.macro
    def quantize_tile(x_ub, workspace_ub, out_ub):
        with T.SimdVF():
            if is_fp4_input:
                zero_bf16 = S.vdup(0.0, T.bfloat16)
            for row in T.serial(block_m):
                for chunk in T.unroll(chunks_per_row, explicit=True):
                    col = chunk * 256
                    expanded = S.vld(
                        workspace_ub[factor_offset + row * subs_per_row + chunk * 8],
                        dist="E2B_B32",
                    )
                    factor_low, factor_high = S.vintlv(expanded, expanded)
                    factors = S.alloc_local((4,), T.float32)
                    factors[0], factors[1] = S.vintlv(factor_low, factor_low)
                    factors[2], factors[3] = S.vintlv(factor_high, factor_high)
                    if is_fp4_input:
                        for pair in T.unroll(2, explicit=True):
                            x_bf16 = S.vcvt(
                                S.vld(x_ub[row, col + pair * 128], dist="UNPK4_B8"),
                                T.bfloat16,
                            )
                            x_low, x_high = S.vintlv(zero_bf16, x_bf16)
                            scaled_low = S.vmul(T.reinterpret(x_low, "float32x64"), factors[pair * 2])
                            scaled_high = S.vmul(T.reinterpret(x_high, "float32x64"), factors[pair * 2 + 1])
                            if is_fp4:
                                low, high = S.vdintlv(
                                    T.reinterpret(scaled_low, "uint16x128"),
                                    T.reinterpret(scaled_high, "uint16x128"),
                                )
                                S.vsts(
                                    out_ub[row, col + pair * 128],
                                    S.vcvt(
                                        T.reinterpret(S.vor(high, S.vmins(low, 1)), "bfloat16x128"),
                                        T.float4_e2m1fn,
                                    ),
                                    dist="PK4_B32",
                                )
                            else:
                                S.vsts(
                                    out_ub[row, col + pair * 128],
                                    S.vcvt(scaled_low, T.float8_e4m3fn),
                                    dist="PK4_B32",
                                )
                                S.vsts(
                                    out_ub[row, col + pair * 128 + 64],
                                    S.vcvt(scaled_high, T.float8_e4m3fn),
                                    dist="PK4_B32",
                                )
                    elif is_fp4:
                        scaled = S.alloc_local((2,), T.float32)
                        for pair in T.unroll(2, explicit=True):
                            for half in T.unroll(2, explicit=True):
                                values = S.vcvt(
                                    S.vld(x_ub[row, col + (pair * 2 + half) * 64], dist="UNPK4_B8"),
                                    T.float32,
                                )
                                scaled[half] = S.vmul(values, factors[pair * 2 + half])
                            low, high = S.vdintlv(
                                T.reinterpret(scaled[0], "uint16x128"),
                                T.reinterpret(scaled[1], "uint16x128"),
                            )
                            S.vsts(
                                out_ub[row, col + pair * 128],
                                S.vcvt(
                                    T.reinterpret(S.vor(high, S.vmins(low, 1)), "bfloat16x128"),
                                    T.float4_e2m1fn,
                                ),
                                dist="PK4_B32",
                            )
                    else:
                        for vector in T.unroll(4, explicit=True):
                            values = S.vcvt(
                                S.vld(x_ub[row, col + vector * 64], dist="UNPK4_B8"),
                                T.float32,
                            )
                            S.vsts(
                                out_ub[row, col + vector * 64],
                                S.vcvt(S.vmul(values, factors[vector]), T.float8_e4m3fn),
                                dist="PK4_B32",
                            )

    @T.macro
    def reorder_input_scales(input_sf_cm_linear_ub, input_sf_linear_ub, row_shift, copied_rows):
        with T.SimdVF():
            lane = S.vci(0, T.int32)
            for input_row in T.serial(input_rows):
                if row_shift is None:
                    index = S.vadds(S.vmuls(lane, input_rows), input_row)
                else:
                    index = S.vadds(S.vmuls(lane, input_rows), T.min(input_row + row_shift, copied_rows - 1))
                S.vsts(
                    input_sf_linear_ub[input_row * input_sf_pitch],
                    S.vgather2(input_sf_cm_linear_ub[0], T.reinterpret(index, "uint32x64")),
                )

    @T.macro
    def reorder_output_scales(workspace_ub, sf_cm_linear_ub):
        with T.SimdVF():
            lane = S.vci(0, T.int32)
            for group in T.serial(groups_per_tile):
                index = S.vadds(S.vmuls(lane, groups_per_tile), group)
                S.vsts(
                    sf_cm_linear_ub[group * block_m],
                    S.vgather2(workspace_ub[0], T.reinterpret(index, "uint32x64")),
                )

    @T.macro
    def decode_input_exponents(input_sf_raw_linear_ub, values_linear_ub, vectors):
        with T.SimdVF():
            for vector in T.serial(vectors):
                exponents = T.reinterpret(
                    S.vld(input_sf_raw_linear_ub[vector * 64], dist="UNPK4_B8"),
                    "uint32x64",
                )
                S.vsts(
                    values_linear_ub[vector * 64],
                    T.reinterpret(S.vshls(exponents, 23), "float32x64"),
                )

    @T.macro
    def reorder_packed_input_scales(input_sf_cm_values_ub, input_sf_linear_ub, row_shift, copied_rows):
        with T.SimdVF():
            lane = S.vci(0, T.int32)
            one = S.vdup(1, T.int32)
            base = S.vadd(S.vmuls(S.vshr(lane, one), input_rows * 2), S.vand(lane, one))
            for input_row in T.serial(input_rows):
                if row_shift is None:
                    source_offset = input_row * 2
                else:
                    source_offset = T.min(input_row + row_shift, copied_rows - 1) * 2
                S.vsts(
                    input_sf_linear_ub[input_row * input_sf_pitch],
                    S.vgather2(
                        input_sf_cm_values_ub[0],
                        T.reinterpret(S.vadds(base, source_offset), "uint32x64"),
                    ),
                )

    @T.macro
    def reorder_packed_output_scales(workspace_ub, sf_bytes_ub):
        with T.SimdVF():
            lane = S.vci(0, T.int32)
            rows_index = S.vmuls(lane, groups_per_tile)
            for pair in T.serial(groups_per_tile // 2):
                first = T.reinterpret(
                    S.vgather2(workspace_ub[0], T.reinterpret(S.vadds(rows_index, pair * 2), "uint32x64")),
                    "uint32x64",
                )
                second = T.reinterpret(
                    S.vgather2(workspace_ub[0], T.reinterpret(S.vadds(rows_index, pair * 2 + 1), "uint32x64")),
                    "uint32x64",
                )
                low, high = S.vintlv(first, second)
                S.vsts(sf_bytes_ub[pair * block_m * 2], low, dist="PK4_B32")
                S.vsts(sf_bytes_ub[pair * block_m * 2 + block_m], high, dist="PK4_B32")

    @T.macro
    def compact_g128_scales(workspace_ub):
        with T.SimdVF():
            lane = S.vci(0, T.int32)
            one = S.vdup(1, T.int32)
            row_shift = S.vdup(groups_per_tile.bit_length() - 1, T.int32)
            chunk_mask = S.vdup(chunks_per_row - 1, T.int32)
            index = S.vadd(
                S.vadd(
                    S.vmuls(S.vshr(lane, row_shift), g128_stage_row_pitch),
                    S.vmuls(S.vand(S.vshr(lane, one), chunk_mask), 8),
                ),
                S.vand(lane, one),
            )
            for batch in T.serial((block_m * groups_per_tile + 63) // 64):
                S.vsts(
                    workspace_ub[batch * 64],
                    S.vgather2(
                        workspace_ub[g128_stage_offset],
                        T.reinterpret(S.vadds(index, batch * (64 // groups_per_tile) * g128_stage_row_pitch), "uint32x64"),
                    ),
                )

    @T.macro
    def pack_row_major_exponents(workspace_ub, sf_bytes_ub):
        with T.SimdVF():
            for batch in T.serial((sf_slots + 63) // 64):
                S.vsts(
                    sf_bytes_ub[batch * 64],
                    T.reinterpret(S.vld(workspace_ub[batch * 64]), "uint32x64"),
                    dist="PK4_B32",
                )

    @T.prim_func
    def per_token_cast_prequant_row_major(
        x: T.StridedTensor[
            (num_tokens, hidden),
            (token_stride, 1),
            input_dtype,
        ],
        x_sf: T.Tensor[x_sf_shape, in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[
            sf_shape,
            (out_sf_stride, 1),
            out_config.sf_dtype,
        ],
    ):
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((block_m, block_k), input_dtype)
            if is_packed_sf:
                if is_col_major_sf:
                    input_sf_raw_ub = T.alloc_shared((input_sf_pitch, input_rows * 2), T.uint8)
                    input_sf_raw_linear_ub = T.Tensor(
                        (input_sf_pitch * input_rows * 2,),
                        T.uint8,
                        input_sf_raw_ub.data,
                    )
                    input_sf_cm_values_ub = T.alloc_shared(
                        (input_sf_pitch * input_rows * 2 + pad,),
                        T.float32,
                    )
                else:
                    input_sf_raw_ub = T.alloc_shared((input_rows + 1, input_sf_pitch), T.uint8)
                    input_sf_raw_linear_ub = T.Tensor(
                        ((input_rows + 1) * input_sf_pitch,),
                        T.uint8,
                        input_sf_raw_ub.data,
                    )
            elif is_col_major_sf:
                input_sf_cm_ub = T.alloc_shared((input_sf_pitch, input_rows), T.float32)
                input_sf_cm_linear_ub = T.Tensor(
                    (input_sf_pitch * input_rows,),
                    T.float32,
                    input_sf_cm_ub.data,
                )
            input_sf_ub = T.alloc_shared((input_rows + 1, input_sf_pitch), T.float32)
            input_sf_linear_ub = T.Tensor(
                ((input_rows + 1) * input_sf_pitch,),
                T.float32,
                input_sf_ub.data,
            )
            if input_group_size == 128:
                input_sf32_ub = T.alloc_shared(((input_rows + 1) * input_sf_pitch,), T.float32)
            workspace_ub = T.alloc_shared((workspace_slots,), T.float32)
            sf_ub = T.Tensor((block_m, groups_per_tile), T.float32, workspace_ub.data)
            out_ub = T.alloc_shared((block_m, block_k), out_config.dtype)
            if is_packed_sf:
                if is_col_major_sf:
                    sf_bytes_ub = T.alloc_shared((groups_per_tile // 2 * block_m * 2,), T.uint8)
                    sf_bytes_view_ub = T.Tensor((groups_per_tile // 2, block_m * 2), T.uint8, sf_bytes_ub.data)
                else:
                    sf_bytes_ub = T.alloc_shared((block_m * groups_per_tile + pad,), T.uint8)
                    sf_bytes_view_ub = T.Tensor((block_m, groups_per_tile), T.uint8, sf_bytes_ub.data)
                T.annotate_buffer_versions(
                    {
                        x_ub: num_stages,
                        input_sf_raw_ub: num_stages,
                        workspace_ub: num_stages,
                        sf_bytes_ub: num_stages,
                        out_ub: num_stages,
                    }
                )
            elif is_col_major_sf:
                sf_cm_ub = T.alloc_shared((groups_per_tile * block_m,), T.float32)
                sf_cm_view_ub = T.Tensor((groups_per_tile, block_m), T.float32, sf_cm_ub.data)
                T.annotate_buffer_versions(
                    {
                        x_ub: num_stages,
                        input_sf_cm_ub: num_stages,
                        workspace_ub: num_stages,
                        sf_cm_ub: num_stages,
                        out_ub: num_stages,
                    }
                )
            else:
                T.annotate_buffer_versions(
                    {
                        x_ub: num_stages,
                        input_sf_ub: num_stages,
                        workspace_ub: num_stages,
                        out_ub: num_stages,
                    }
                )

            for token_tile, hidden_tile in T.Persistent(
                [T.ceildiv(num_tokens, block_m), num_hidden_tiles],
                num_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                token_base = token_tile * block_m
                hidden_base = hidden_tile * block_k
                sf_base = hidden_tile * groups_per_tile
                valid_rows = T.min(block_m, num_tokens - token_base)
                T.copy(
                    x[
                        token_base : token_base + valid_rows,
                        hidden_base : hidden_base + block_k,
                    ],
                    x_ub[:valid_rows, :block_k],
                )
                input_row_base = token_base // input_block_m
                input_group_base = hidden_base // input_group_size
                if is_packed_sf:
                    if is_col_major_sf:
                        if input_block_m <= block_m:
                            input_sf_total_rows = T.ceildiv(num_tokens, input_block_m)
                            copied_input_rows = T.min(input_rows, input_sf_total_rows)
                            input_copy_base = T.max(0, T.min(input_row_base, input_sf_total_rows - copied_input_rows))
                            T.copy(
                                x_sf[
                                    input_group_base // 2 : input_group_base // 2 + input_groups_per_tile // 2,
                                    input_copy_base * 2 : input_copy_base * 2 + copied_input_rows * 2,
                                ],
                                input_sf_raw_ub[: input_groups_per_tile // 2, : copied_input_rows * 2],
                            )
                            decode_input_exponents(input_sf_raw_linear_ub, input_sf_cm_values_ub, input_rows * 2)
                            reorder_packed_input_scales(
                                input_sf_cm_values_ub,
                                input_sf_linear_ub,
                                input_row_base - input_copy_base,
                                copied_input_rows,
                            )
                        else:
                            T.copy(
                                x_sf[
                                    input_group_base // 2 : input_group_base // 2 + input_groups_per_tile // 2,
                                    input_row_base * 2 : input_row_base * 2 + 2,
                                ],
                                input_sf_raw_ub[: input_groups_per_tile // 2, :2],
                            )
                            decode_input_exponents(input_sf_raw_linear_ub, input_sf_cm_values_ub, input_rows * 2)
                            reorder_packed_input_scales(input_sf_cm_values_ub, input_sf_linear_ub, None, None)
                    else:
                        if input_block_m <= block_m:
                            valid_input_rows = T.ceildiv(valid_rows, input_block_m)
                            T.copy(
                                x_sf[
                                    input_row_base : input_row_base + valid_input_rows,
                                    input_group_base : input_group_base + input_groups_per_tile,
                                ],
                                input_sf_raw_ub[:valid_input_rows, :input_groups_per_tile],
                            )
                        else:
                            T.copy(
                                x_sf[
                                    input_row_base : input_row_base + 1,
                                    input_group_base : input_group_base + input_groups_per_tile,
                                ],
                                input_sf_raw_ub[:1, :input_groups_per_tile],
                            )
                        decode_input_exponents(input_sf_raw_linear_ub, input_sf_linear_ub, input_rows)
                elif is_col_major_sf:
                    if input_block_m <= block_m:
                        input_sf_total_rows = T.ceildiv(num_tokens, input_block_m)
                        copied_input_rows = T.min(input_rows, input_sf_total_rows)
                        input_copy_base = T.max(0, T.min(input_row_base, input_sf_total_rows - copied_input_rows))
                        T.copy(
                            x_sf[
                                input_group_base : input_group_base + input_groups_per_tile,
                                input_copy_base : input_copy_base + copied_input_rows,
                            ],
                            input_sf_cm_ub[:input_groups_per_tile, :copied_input_rows],
                        )
                        reorder_input_scales(
                            input_sf_cm_linear_ub,
                            input_sf_linear_ub,
                            input_row_base - input_copy_base,
                            copied_input_rows,
                        )
                    else:
                        T.copy(
                            x_sf[
                                input_group_base : input_group_base + input_groups_per_tile,
                                input_row_base : input_row_base + 1,
                            ],
                            input_sf_cm_ub[:input_groups_per_tile, :1],
                        )
                        reorder_input_scales(input_sf_cm_linear_ub, input_sf_linear_ub, None, None)
                else:
                    if input_block_m <= block_m:
                        valid_input_rows = T.ceildiv(valid_rows, input_block_m)
                        T.copy(
                            x_sf[
                                input_row_base : input_row_base + valid_input_rows,
                                input_group_base : input_group_base + input_groups_per_tile,
                            ],
                            input_sf_ub[:valid_input_rows, :input_groups_per_tile],
                        )
                    else:
                        T.copy(
                            x_sf[
                                input_row_base : input_row_base + 1,
                                input_group_base : input_group_base + input_groups_per_tile,
                            ],
                            input_sf_ub[:1, :input_groups_per_tile],
                        )
                if input_group_size == 128:
                    expand_input_scales(input_sf_linear_ub, input_sf32_ub)
                    reduce_and_scale(
                        x_ub,
                        input_sf32_ub,
                        workspace_ub,
                        sf_bytes_ub if is_packed_sf and not is_col_major_sf else None,
                    )
                else:
                    reduce_and_scale(
                        x_ub,
                        input_sf_linear_ub,
                        workspace_ub,
                        sf_bytes_ub if is_packed_sf and not is_col_major_sf else None,
                    )
                if group_size == 128:
                    compact_g128_scales(workspace_ub)
                if is_packed_sf:
                    if is_col_major_sf:
                        reorder_packed_output_scales(workspace_ub, sf_bytes_ub)
                        T.copy(
                            sf_bytes_view_ub[: groups_per_tile // 2, : valid_rows * 2],
                            out_sf[
                                sf_base // 2 : sf_base // 2 + groups_per_tile // 2,
                                token_base * 2 : token_base * 2 + valid_rows * 2,
                            ],
                        )
                    else:
                        pack_row_major_exponents(workspace_ub, sf_bytes_ub)
                        T.copy(
                            sf_bytes_view_ub[:valid_rows, :groups_per_tile],
                            out_sf[
                                token_base : token_base + valid_rows,
                                sf_base : sf_base + groups_per_tile,
                            ],
                        )
                elif is_col_major_sf:
                    reorder_output_scales(workspace_ub, sf_cm_ub)
                    T.copy(
                        sf_cm_view_ub[:groups_per_tile, :valid_rows],
                        out_sf[
                            sf_base : sf_base + groups_per_tile,
                            token_base : token_base + valid_rows,
                        ],
                    )
                else:
                    T.copy(
                        sf_ub[:valid_rows, :groups_per_tile],
                        out_sf[
                            token_base : token_base + valid_rows,
                            sf_base : sf_base + groups_per_tile,
                        ],
                    )
                quantize_tile(x_ub, workspace_ub, out_ub)
                T.copy(
                    out_ub[:valid_rows, :block_k],
                    out[
                        token_base : token_base + valid_rows,
                        hidden_base : hidden_base + block_k,
                    ],
                )

    return per_token_cast_prequant_row_major
