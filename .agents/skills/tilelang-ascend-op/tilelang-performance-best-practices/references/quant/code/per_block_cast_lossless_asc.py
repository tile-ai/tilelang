import math

import tilelang
from tilelang import language as T
from tilelang.language import simd as S

from tile_kernels.config import get_num_vec_cores
from tile_kernels.quant.common import *


def vector_mask(dtype, num_lanes):
    return S.pset(dtype.bits, 'PAT_VL' + str(num_lanes))


def lane_indices(dtype):
    index_dtype, vector_type = {8: (T.int8, 'uint8x256'), 16: (T.int16, 'uint16x128'), 32: (T.int32, 'uint32x64')}[dtype.bits]
    return T.reinterpret(S.vci(0, index_dtype), vector_type)


def get_sf_storage(config, dtype):
    if config.use_packed_ue8m0:
        return T.uint16, get_packed_ue8m0_pack_factor()
    return dtype, 1


@T.macro
def init_fp4_to_fp8_map(fp4_to_fp8_map_ub, invalid_exp_ub):
    with T.SimdVF():
        # The mapping index stores exp_delta in the high bits and fp4_code in the low 4 bits;
        # each entry contains the equivalent FP8 bit pattern.
        lanes = lane_indices(T.int8)
        fp8_base_u32 = S.alloc_local((2,), T.uint32)
        fp8_base_u32[0], fp8_base_u32[1] = S.vintlv(S.vdup(T.uint32(0x3C383000), T.uint32), S.vdup(T.uint32(0x4C484440), T.uint32))
        code = S.vand(lanes, S.vdup(T.uint8(0x0F), T.uint8))
        magnitude = S.vand(code, S.vdup(T.uint8(0x07), T.uint8))
        base = S.vselr(T.reinterpret(fp8_base_u32[0], 'uint8x256'), magnitude)
        delta = S.vadds(S.vand(S.vshrs(lanes, 1), S.vdup(T.uint8(0x78), T.uint8)), T.uint8(216))
        scaled = S.vadd(base, delta)
        nonzero_mask = S.vcmps(magnitude, T.uint8(0), op='ne')
        nonzero = S.vsel(scaled, S.vdup(T.uint8(0), T.uint8), nonzero_mask)
        S.vsts(fp4_to_fp8_map_ub[0], S.vor(nonzero, S.vshls(S.vand(code, S.vdup(T.uint8(0x08), T.uint8)), 4)), dist='NORM_B8')
        assert_mask = vector_mask(T.uint16, 16)
        S.vsts(invalid_exp_ub[0], S.vdup(0, T.uint16), assert_mask, dist='NORM_B16')


@tilelang.jit()
def get_per_block_cast_lossless_kernel_asc(hidden: int, in_config: CastInputConfig, out_config: CastOutputConfig):
    assert in_config.dtype == T.float4_e2m1fn and out_config.dtype == T.float8_e4m3fn
    assert in_config.sf_block == (1, 32)
    assert out_config.sf_block in ((32, 32), (1, 128), (128, 128))
    assert out_config.round_sf
    assert not in_config.use_tma_aligned_col_major_sf or in_config.use_packed_ue8m0
    assert not out_config.use_tma_aligned_col_major_sf or out_config.use_packed_ue8m0
    assert hidden % out_config.sf_block[1] == 0
    # TMA (col-major) output sf is only supported for the block-level (32, 32) layout:
    # per-data-row sf (1, 128) and the single-sf-per-block (128, 128) case cannot be
    # packed into UE8M0 within one block (tests filter those combinations on Ascend).
    assert not out_config.use_tma_aligned_col_major_sf or out_config.sf_block == (32, 32)

    out_sf_m, out_sf_k = out_config.sf_block
    if out_sf_m == 1:
        # (1, 128): every data row produces its own out-sf. block_m=64 halves the
        # token-direction tile count (8064/32=252 -> 126 tiles) and cuts per-tile
        # fixed overhead (kernel loop / T.copy launch / quant prologue) in half.
        # UB stays well inside budget (~42KB/stage vs 248KB). out_sf_m != 1 uses
        # out_sf_m rows per tile (128) which is already UB-bound for block_k=128.
        block_m = 64
    else:
        block_m = out_sf_m
    if out_sf_k == 128:
        # Output sf covers 128 columns; fix block_k at 128 so the 4-slot column
        # merge reduces to a single PAT_ALL group (verified) and every block
        # produces exactly one out-sf column. block_k = 128 also keeps block_m =
        # 128 (i.e. (128, 128)) inside the shared-memory budget.
        block_k = 128
    else:
        block_k = math.gcd(2048, hidden)
    assert block_k % out_sf_k == 0 and hidden % block_k == 0
    if block_k <= 1024 and block_m <= 64:
        num_stages = 3
    else:
        num_stages = 2
    num_per_channels = in_config.sf_block[1]
    num_sf_cols_per_block = block_k // num_per_channels
    vec_size = 64
    assert vec_size * num_per_channels >= block_k

    num_cores = get_num_vec_cores()
    sf_dtype, in_pack_factor = get_sf_storage(in_config, T.uint32)
    num_sf_load_cols = num_sf_cols_per_block // in_pack_factor
    out_sf_dtype, out_pack_factor = get_sf_storage(out_config, T.float32)
    # Output sf granularity is out_sf_k columns; the reduce pipeline works on
    # 32-column slots and stores only one slot per out sf group.
    num_out_sf_cols_per_block = block_k // out_sf_k
    if out_config.use_packed_ue8m0:
        num_sf_store_cols = ceil_div(num_out_sf_cols_per_block, out_pack_factor)
    else:
        num_sf_store_cols = num_out_sf_cols_per_block
    num_out_sf_rows = block_m if out_sf_m == 1 else 1
    if in_config.use_tma_aligned_col_major_sf:
        sf_load_ub_shape = (num_sf_load_cols, block_m * in_pack_factor)
    else:
        sf_load_ub_shape = (block_m, vec_size * in_pack_factor)
    if out_config.use_tma_aligned_col_major_sf:
        num_batched_tiles = 2
        out_sf_shape_ub = (max(num_sf_store_cols, 4 * out_pack_factor), num_batched_tiles * out_pack_factor)
    else:
        num_batched_tiles = 1
        out_sf_shape_ub = (num_out_sf_rows, max(num_sf_store_cols, 8 * out_pack_factor) * out_pack_factor)
    num_tokens = T.dynamic('num_tokens')
    in_sf_shape = get_sf_shape((num_tokens, hidden), in_config)
    out_sf_shape = get_sf_shape((num_tokens, hidden), out_config)
    if in_config.use_tma_aligned_col_major_sf:
        in_sf_stride = T.dynamic('in_sf_stride')
    else:
        in_sf_stride = in_sf_shape[1]
    if out_config.use_tma_aligned_col_major_sf:
        out_sf_stride = T.dynamic('out_sf_stride')
    else:
        out_sf_stride = out_sf_shape[1]

    # For block_k <= 256 the quant tile always runs on 256-column tiles; x_ub/out_ub
    # are padded to 256 columns (zero-filled) so a single 256-column code path covers
    # block_k in {64, 128, 256}.
    quant_cols = 256 if block_k <= 256 else block_k

    @T.macro
    def load_sf(sf_load_ub, row, gather_base, gather_mask):
        if in_config.use_tma_aligned_col_major_sf:
            values = S.vgather2(sf_load_ub[0, 0], S.vadds(gather_base, row * in_pack_factor), gather_mask)
        elif sf_dtype == T.uint16:
            values = S.vld(sf_load_ub[row, 0], dist='UNPK_B8')
        else:
            values = S.vld(sf_load_ub[row, 0], dist='NORM')
        if sf_dtype == T.uint16:
            return T.reinterpret(values, 'uint16x128')
        return S.vpack(S.vshrs(T.reinterpret(values, 'uint32x64'), 23))

    @T.macro
    def copy_sf(x_sf, sf_load_ub, row_start, pid_k):
        if in_config.use_tma_aligned_col_major_sf:
            col_start = T.alloc_var(T.int32, init=pid_k * num_sf_load_cols)
            T.copy(x_sf[col_start, row_start * in_pack_factor], sf_load_ub[:num_sf_load_cols, : block_m * in_pack_factor])
        else:
            col_start = T.alloc_var(T.int32, init=pid_k * num_sf_cols_per_block)
            T.copy(x_sf[row_start, col_start], sf_load_ub[:block_m, :num_sf_cols_per_block])

    @T.macro
    def col_merge(v):
        # Merge every 4 consecutive 32-column slots into a single 128-column
        # out-sf value. block_k is fixed at 128 for out_sf_k == 128, so the merge
        # is a single PAT_ALL group reduce + broadcast (verified semantics).
        pat_all = S.pset(16, 'PAT_ALL')
        return S.vdupv(S.vcmax(v, pat_all), pat_all)

    @T.macro
    def emit_out_sf(sf_exp_ub, out_sf_ub, invalid_exp_ub, out_exp_row, min_exp_row, sf_exp_row, sf_ub_row, store_row, mask, zero, six):
        full_mask = vector_mask(T.uint16, max(num_sf_cols_per_block, 16))
        S.vsts(sf_exp_ub[sf_exp_row, 0], S.vadds(out_exp_row, -5), full_mask, dist='NORM_B16')
        invalid = S.vsub(out_exp_row, S.vmin(out_exp_row, S.vadds(min_exp_row, 5)))
        max_invalid = S.vcmax(invalid, mask)
        acc_invalid = S.vmax(max_invalid, S.vld(invalid_exp_ub[0], dist='BRC_B16'))
        assert_mask = vector_mask(T.uint16, 16)
        S.vsts(invalid_exp_ub[0], acc_invalid, assert_mask, dist='NORM_B16')
        out_sf_value = S.alloc_var(out_sf_dtype)
        if out_sf_dtype == T.uint16:
            out_sf_value = T.reinterpret(S.vpack(out_exp_row), 'uint16x128')
        else:
            wide_exp, _ = S.vintlv(out_exp_row, zero)
            out_sf_value = T.reinterpret(S.vshls(T.reinterpret(wide_exp, 'uint32x64'), 23), 'float32x64')
        out_store_mask = vector_mask(out_sf_dtype, num_sf_store_cols)
        if out_config.use_tma_aligned_col_major_sf:
            store_indices = lane_indices(out_sf_dtype)
            offsets = S.vadds(S.vmuls(store_indices, num_batched_tiles), store_row)
            S.vscatter(out_sf_value, out_sf_ub[0, 0], offsets, out_store_mask)
        elif out_sf_dtype == T.uint16:
            S.vsts(out_sf_ub[sf_ub_row, 0], out_sf_value, out_store_mask, dist='NORM_B16')
        else:
            S.vsts(out_sf_ub[sf_ub_row, 0], out_sf_value, out_store_mask, dist='NORM_B32')

    @T.macro
    def reduce_sf(sf_load_ub, sf_exp_ub, out_sf_ub, invalid_exp_ub, store_row):
        with T.SimdVF():
            mask = vector_mask(T.uint16, num_sf_cols_per_block)
            full_mask = vector_mask(T.uint16, max(num_sf_cols_per_block, 16))
            zero = S.vdup(0, T.uint16)
            six = S.vdup(6, T.uint16)
            gather_base = S.alloc_var(sf_dtype)
            gather_mask = vector_mask(in_config.sf_dtype, num_sf_cols_per_block * in_pack_factor)
            if in_config.use_tma_aligned_col_major_sf:
                lanes = lane_indices(sf_dtype)
                if in_config.use_packed_ue8m0:
                    pair_index = S.vshrs(lanes, 1)
                    byte_index = S.vand(lanes, S.vdup(1, T.uint16))
                    gather_base = S.vadd(S.vmuls(pair_index, block_m * in_pack_factor), byte_index)
                else:
                    gather_base = S.vmuls(lanes, block_m)

            max_exp0 = S.alloc_var(T.uint16)
            max_exp1 = S.alloc_var(T.uint16)
            min_exp0 = S.alloc_var(T.uint16)
            min_exp1 = S.alloc_var(T.uint16)
            max_exp0 = zero
            max_exp1 = zero
            min_exp0 = S.vdup(T.max_value(T.uint16), T.uint16)
            min_exp1 = S.vdup(T.max_value(T.uint16), T.uint16)
            for row in T.serial(0, block_m, 2):
                in_exp0 = load_sf(sf_load_ub, row, gather_base, gather_mask)
                in_exp0_clean = S.vsel(in_exp0, zero, mask)
                S.vsts(sf_exp_ub[row, 0], in_exp0_clean, full_mask, dist='NORM_B16')
                in_exp1 = load_sf(sf_load_ub, row + 1, gather_base, gather_mask)
                in_exp1_clean = S.vsel(in_exp1, zero, mask)
                S.vsts(sf_exp_ub[row + 1, 0], in_exp1_clean, full_mask, dist='NORM_B16')
                if out_sf_m == 1:
                    # (1, 128): every data row produces its own out-sf row.
                    # max_exp/min_exp cross-row reduction is only needed for the
                    # block-level (out_sf_m != 1) path, so skip it here to save 4
                    # vector ops per 2 rows (compile-time branch).
                    out_exp0 = col_merge(S.vsub(S.vmax(in_exp0_clean, six), six))
                    out_exp1 = col_merge(S.vsub(S.vmax(in_exp1_clean, six), six))
                    emit_out_sf(sf_exp_ub, out_sf_ub, invalid_exp_ub, out_exp0, in_exp0_clean, block_m + row, row, store_row, mask, zero, six)
                    emit_out_sf(sf_exp_ub, out_sf_ub, invalid_exp_ub, out_exp1, in_exp1_clean, block_m + row + 1, row + 1, store_row, mask, zero, six)
                else:
                    max_exp0 = S.vmax(max_exp0, in_exp0_clean, mask)
                    min_exp0 = S.vmin(min_exp0, in_exp0_clean, mask)
                    max_exp1 = S.vmax(max_exp1, in_exp1_clean, mask)
                    min_exp1 = S.vmin(min_exp1, in_exp1_clean, mask)
            if out_sf_m != 1:
                max_exp = S.vmax(max_exp0, max_exp1, mask)
                min_exp = S.vmin(min_exp0, min_exp1, mask)
                out_exp = S.alloc_var(T.uint16)
                out_exp = S.vsub(S.vmax(max_exp, six), six)
                if out_sf_k == 128:
                    out_exp = col_merge(out_exp)
                emit_out_sf(sf_exp_ub, out_sf_ub, invalid_exp_ub, out_exp, min_exp, block_m, 0, store_row, mask, zero, six)

    @T.macro
    def quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8, out_exp, row, chunk):
        in_exp = S.vld(sf_exp_ub[row, chunk * 8], dist='E2B_B16')
        delta = S.vmuls(S.vsub(in_exp, out_exp), T.uint16(0x1010))
        index = S.vadd(fp4_codes_u8, T.reinterpret(delta, 'uint8x256'))
        quantized = T.reinterpret(S.vselr(fp4_to_fp8_map, index), 'float8_e4m3fnx256')
        S.vsts(out_ub[row, chunk * 256], quantized, dist='NORM_B8')

    @T.macro
    def quant(sf_exp_ub, x_ub, out_ub, fp4_to_fp8_map_ub):
        with T.SimdVF():
            mask = S.vdup(T.uint8(0x0F), T.uint8)
            fp4_to_fp8_map = S.vld(fp4_to_fp8_map_ub[0])
            if block_k <= 256:
                # x_ub/out_ub are padded to 256 columns; the 256-column tile reads two
                # rows and writes the effective block_k columns (trailing padded code
                # is zero so the padded output columns hold valid FP8 zeros that get
                # truncated on store-back).
                for row in T.serial(0, block_m, 2):
                    if out_sf_m == 1:
                        out_exp0 = S.vld(sf_exp_ub[block_m + row, 0], dist='E2B_B16')
                        out_exp1 = S.vld(sf_exp_ub[block_m + row + 1, 0], dist='E2B_B16')
                    else:
                        out_exp0 = S.vld(sf_exp_ub[block_m, 0], dist='E2B_B16')
                        out_exp1 = out_exp0
                    packed = T.reinterpret(S.vld(x_ub[row, 0]), 'uint8x256')
                    fp4_codes_u8_0, fp4_codes_u8_1 = S.vintlv(S.vand(packed, mask), S.vshrs(packed, 4))
                    quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8_0, out_exp0, row, 0)
                    quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8_1, out_exp1, row + 1, 0)
            else:
                for chunk in T.serial(0, block_k // 256, 2):
                    for row in T.serial(0, block_m, 2):
                        if out_sf_m == 1:
                            out_exp0 = S.vld(sf_exp_ub[block_m + row, chunk * 8], dist='E2B_B16')
                            out_exp1 = S.vld(sf_exp_ub[block_m + row, (chunk + 1) * 8], dist='E2B_B16')
                        else:
                            out_exp0 = S.vld(sf_exp_ub[block_m, chunk * 8], dist='E2B_B16')
                            out_exp1 = S.vld(sf_exp_ub[block_m, (chunk + 1) * 8], dist='E2B_B16')
                        packed0 = T.reinterpret(S.vld(x_ub[row, chunk * 256]), 'uint8x256')
                        packed1 = T.reinterpret(S.vld(x_ub[row + 1, chunk * 256]), 'uint8x256')
                        fp4_codes_u8_0_0, fp4_codes_u8_0_1 = S.vintlv(S.vand(packed0, mask), S.vshrs(packed0, 4))
                        fp4_codes_u8_1_0, fp4_codes_u8_1_1 = S.vintlv(S.vand(packed1, mask), S.vshrs(packed1, 4))
                        quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8_0_0, out_exp0, row, chunk)
                        quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8_0_1, out_exp1, row, chunk + 1)
                        quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8_1_0, out_exp0, row + 1, chunk)
                        quant_chunk(sf_exp_ub, out_ub, fp4_to_fp8_map, fp4_codes_u8_1_1, out_exp1, row + 1, chunk + 1)

    @T.prim_func
    def kernel(
        x: T.Tensor[(num_tokens, hidden), in_config.dtype],
        x_sf: T.StridedTensor[in_sf_shape, (in_sf_stride, 1), in_config.sf_dtype],
        out: T.Tensor[(num_tokens, hidden), out_config.dtype],
        out_sf: T.StridedTensor[out_sf_shape, (out_sf_stride, 1), out_config.sf_dtype],
    ):
        with T.Kernel(num_cores) as core_id:
            x_ub = T.alloc_shared((block_m, quant_cols), in_config.dtype)
            sf_exp_ub = T.alloc_shared((block_m + num_out_sf_rows, max(num_sf_cols_per_block, 16)), T.uint16)
            out_ub = T.alloc_shared((block_m, quant_cols), out_config.dtype)
            fp4_to_fp8_map_ub = T.alloc_shared((256,), T.uint8)
            sf_load_ub = T.alloc_shared(sf_load_ub_shape, in_config.sf_dtype)
            out_sf_ub = T.alloc_shared(out_sf_shape_ub, out_config.sf_dtype)
            invalid_exp_ub = T.alloc_shared((16,), T.uint16)
            T.annotate_buffer_versions({
                x_ub: num_stages,
                sf_load_ub: num_stages,
                sf_exp_ub: num_stages,
                out_ub: num_stages,
                out_sf_ub: num_stages,
                invalid_exp_ub: 1,
            })
            init_fp4_to_fp8_map(fp4_to_fp8_map_ub, invalid_exp_ub)
            for pid_m, pid_k in T.Persistent(
                [num_tokens // (block_m * num_batched_tiles), hidden // block_k],
                num_cores,
                core_id,
                group_size=1,
                num_stages=num_stages,
            ):
                for batch in T.serial(num_batched_tiles):
                    tile_m = T.alloc_var(T.int32, init=pid_m * num_batched_tiles + batch)
                    row_start = T.alloc_var(T.int32, init=tile_m * block_m)
                    if block_k < 256:
                        # Zero-fill the padded 256-column tile so the padded FP4 bytes
                        # decode to code 0 (legal lookup) during quant. S.vsts writes
                        # 256 bytes (two 128-byte rows) per call.
                        with T.SimdVF():
                            zero8 = S.vdup(0, T.uint8)
                            for pad_row in T.serial(0, block_m, 2):
                                S.vsts(x_ub[pad_row, 0], zero8, dist='NORM_B8')
                    T.copy(x[row_start, pid_k * block_k], x_ub[:block_m, :block_k])
                    copy_sf(x_sf, sf_load_ub, row_start, pid_k)
                    reduce_sf(sf_load_ub, sf_exp_ub, out_sf_ub, invalid_exp_ub, batch)
                    quant(sf_exp_ub, x_ub, out_ub, fp4_to_fp8_map_ub)
                    T.copy(out_ub[:block_m, :block_k], out[row_start, pid_k * block_k])
                first_tile = T.alloc_var(T.int32, init=pid_m * num_batched_tiles)
                store_base = T.alloc_var(T.int32, init=pid_k * num_sf_store_cols)
                if out_config.use_tma_aligned_col_major_sf:
                    T.copy(out_sf_ub[:num_sf_store_cols, : num_batched_tiles * out_pack_factor], out_sf[store_base, first_tile * out_pack_factor])
                else:
                    # out_sf_ub stores one sf per 32-column slot; its copy width is in
                    # storage elements (uint8 for packed UE8M0, otherwise sf dtype).
                    store_copy_cols = num_sf_store_cols * out_pack_factor if out_config.use_packed_ue8m0 else num_sf_store_cols
                    T.copy(out_sf_ub[:num_out_sf_rows, :store_copy_cols], out_sf[first_tile * num_out_sf_rows, store_base * out_pack_factor])
            T.device_assert(invalid_exp_ub[0] == 0)

    return kernel
