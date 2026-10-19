"""SIMD scale packing, layout conversion and psum padding for MXFP8 GEMM."""

import tilelang
import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S

from ..config import GemmType, Major


@T.macro
def _transpose_16rows(src, dst, row, cols, stride, mask):
    # Four interleave levels transpose 16 rows; scatter stores emit 16-element columns.
    num_regs = cols // 8
    regs = S.alloc_local((num_regs,), "uint16")
    src_ptr = S.make_ubuf_ptr(T.access_ptr(src[row, 0], "r", extent=16 * cols), "uint16")
    for i in T.Unroll(num_regs, explicit=True):
        regs[i], src_ptr = S.vld(src_ptr, post_inc=128)
    for level in T.Unroll(4, explicit=True):
        offset = (num_regs // 2) >> (level % (num_regs.bit_length() - 1))
        for pair in T.Unroll(num_regs // 2, explicit=True):
            a = (pair // offset) * (2 * offset) + pair % offset
            b = a + offset
            regs[a], regs[b] = S.vintlv(regs[a], regs[b])
    dst_ptr = S.make_ubuf_ptr(T.access_ptr(dst[0, row], "w", extent=(cols - 1) * stride + 16), "uint16")
    for i in T.Unroll(num_regs, explicit=True):
        idx = (i // 2 + (i % 2) * 4) if cols == 64 else i
        dst_ptr = S.vsstb(regs[idx], dst_ptr, T.int32(((stride // 16) << 16) | (stride // 2)), mask, update=True)


@tilelang.jit
def build_transform_sf(is_float, major, config, gemm_type=GemmType.Normal, alignment=256, gran_mn=1):
    src_block_m, src_block_k = config.src_block_m, config.src_block_k
    dtype = "float32" if is_float else "int16"
    num_groups = T.dynamic("num_groups")
    m_grouped = gemm_type == GemmType.MGroupedContiguousWithPsumLayout
    k_grouped = gemm_type == GemmType.KGroupedContiguousWithPsumLayout
    grouped = m_grouped or k_grouped
    batches = T.dynamic("batches")
    src_rows, src_cols = T.dynamic("src_rows"), T.dynamic("src_cols")
    mn, pairs = T.dynamic("mn"), T.dynamic("pairs")
    strides = (T.dynamic("batch_stride", dtype="int64"), T.dynamic("outer_stride", dtype="int64"), 1)
    src_mn, src_k = (src_rows, src_cols) if major == Major.K else (src_cols, src_rows)
    pack, bits, raw_dtype = (2, 32, "uint32") if is_float else (1, 16, "uint16")
    lanes = 2048 // bits
    block_mn, block_pairs = src_block_m * gran_mn, src_block_k // pack
    ub_shape = (max(16, src_block_m), src_block_k) if major == Major.K else (src_block_k, max(128, src_block_m))
    converted_stride = (max(16, src_block_m) + 16) if major == Major.K else max(128, src_block_m)
    group_extent = src_block_m if m_grouped else src_block_k
    group_divisor = gran_mn if m_grouped else (32 if is_float else 64)
    assert not grouped or alignment % group_divisor == 0
    clear_rows = m_grouped and major == Major.K and block_mn <= alignment
    scalar_k_mask = k_grouped and major == Major.MN and src_block_k <= alignment // group_divisor
    # Each nonempty group occupies a distinct aligned interval.
    max_group_ranges = (group_extent * group_divisor + alignment - 1) // alignment + 1
    gran_shift = gran_mn.bit_length() - 1
    repeat_regs = min(max(1, gran_mn // 16), block_mn // 128)
    # Leave 8 KiB for compiler reservations, metadata and alignment.
    source_bytes = ub_shape[0] * ub_shape[1] * (bits // 8)
    packed_bytes_per_tile = max(16, src_block_m) * block_pairs * 2 if major == Major.K and is_float else 0
    tile_bytes = source_bytes + packed_bytes_per_tile + block_pairs * converted_stride * 2
    if gran_mn != 1:
        tile_bytes += block_pairs * block_mn * 2
    mask_bytes = ((group_extent + lanes - 1) // lanes) * lanes * (bits // 8) if grouped else 0
    preferred_stages = 2 if major == Major.K and is_float and gran_mn == 1 or major == Major.MN and gran_mn == 128 else 3
    range_bytes = ((max_group_ranges * 8 + 31) // 32) * 32 if grouped and not (clear_rows or scalar_k_mask) else 0
    capacity = (248 * 1024 - mask_bytes - range_bytes) // tile_bytes
    num_stages = min(preferred_stages, capacity)
    assert num_stages > 0, "transform_sf tile exceeds UB capacity"
    versions = (num_stages, "counter")

    @T.prim_func
    def main(
        source: T.StridedTensor((batches, src_rows, src_cols), strides, dtype),
        groups: T.Tensor((num_groups,), "int32"),
        output: T.Tensor((batches, pairs, mn), "int16"),
    ):
        with T.Kernel(config.num_blocks) as core:
            T.assume(batches > 0 and src_rows > 0 and src_cols > 0 and mn > 0 and pairs > 0)
            source_ub = T.alloc_shared(ub_shape, dtype)
            source_bits = T.view(source_ub, dtype=raw_dtype)
            if major == Major.K and not is_float:
                packed_ub = T.view(source_ub, dtype="uint16")
            else:
                packed_ub = T.alloc_shared((max(16, src_block_m), block_pairs), "uint16")
            packed_bytes = T.view(packed_ub, shape=(max(16, src_block_m), block_pairs * 2), dtype="uint8")
            converted = T.alloc_shared((block_pairs, converted_stride), "uint16")
            if gran_mn == 1:
                output_ub = T.view(converted, dtype="int16")
            else:
                output_ub = T.alloc_shared((block_pairs, block_mn), "int16")
            output_bits = T.view(output_ub, dtype="uint16")
            validation_ub = T.alloc_shared((64,), "uint32")
            group_mask_ub = T.alloc_shared((T.ceildiv(group_extent, lanes) * lanes,), raw_dtype)
            group_ranges_ub = T.alloc_shared((max_group_ranges, 2), "int32")
            T.annotate_buffer_versions(
                {
                    source_ub: versions,
                    packed_ub: versions if major == Major.K else 1,
                    converted: versions,
                    output_ub: versions,
                    validation_ub: 1,
                    group_mask_ub: 1,
                    group_ranges_ub: 1,
                }
            )
            if is_float:
                with T.SimdVF():
                    S.vsts(validation_ub[0], S.vdup(0, "uint32"))
            tiles_mn, tiles_k = T.ceildiv(mn, block_mn), T.ceildiv(pairs, block_pairs)
            num_tiles = batches * tiles_mn * tiles_k
            count = T.ceildiv(num_tiles, config.num_blocks)
            for tile in T.Pipelined(core * count, T.min((core + 1) * count, num_tiles), num_stages=num_stages):
                with T.Stage(0):
                    T.assume(tile >= 0)
                    # Distinct tile indices write disjoint GM regions.
                    T.assume_no_conflict(output, level=0, cross=True)
                    batch = tile // (tiles_mn * tiles_k)
                    mn_idx = (tile % tiles_mn if k_grouped else tile // tiles_k % tiles_mn) * block_mn
                    pair_idx = (tile // tiles_mn % tiles_k if k_grouped else tile % tiles_k) * block_pairs
                    src_mn_idx, src_k_idx = mn_idx // gran_mn, pair_idx * pack
                    actual_mn = T.min(block_mn, mn - mn_idx)
                    actual_pairs = T.min(block_pairs, pairs - pair_idx)
                    actual_src_mn = T.min(src_block_m, src_mn - src_mn_idx)
                    if not is_float:
                        actual_src_mn = T.alloc_var("int32", init=actual_src_mn)
                    actual_src_k = T.min(src_block_k, src_k - src_k_idx)
                    if not is_float:
                        actual_src_k = T.alloc_var("int32", init=actual_src_k)
                    if major == Major.K:
                        T.copy(
                            source[batch, src_mn_idx : src_mn_idx + actual_src_mn, src_k_idx : src_k_idx + actual_src_k],
                            source_ub[:actual_src_mn, :actual_src_k],
                            l2_cache_ctrl=config.l2_ctrl_load,
                        )
                    else:
                        T.copy(
                            source[batch, src_k_idx : src_k_idx + actual_src_k, src_mn_idx : src_mn_idx + actual_src_mn],
                            source_ub[:actual_src_k, :actual_src_mn],
                            l2_cache_ctrl=config.l2_ctrl_load,
                        )
                    if grouped:
                        base = src_mn_idx if m_grouped else src_k_idx
                        first = T.alloc_var("int32", init=0)
                        last = T.alloc_var("int32", init=num_groups)
                        while first < last:
                            mid = (first + last) // 2
                            if groups[mid] <= base * group_divisor:
                                first = mid + 1
                            else:
                                last = mid
                    if clear_rows or scalar_k_mask:
                        lo = first
                        group_actual = actual_src_mn if clear_rows else actual_src_k
                        group_base = src_mn_idx if clear_rows else src_k_idx
                        lower = T.alloc_var("int32", init=0)
                        upper = T.alloc_var("int32", init=0)
                        if lo < num_groups:
                            if lo > 0:
                                lower = T.max(
                                    0, T.min(group_actual, T.ceildiv(groups[lo - 1], alignment) * (alignment // group_divisor) - group_base)
                                )
                            upper = T.max(0, T.min(group_actual, T.ceildiv(groups[lo], group_divisor) - group_base))
                    if grouped and not (clear_rows or scalar_k_mask):
                        num_ranges = T.alloc_var("int32", init=0)
                        lo = T.alloc_var("int32", init=first)
                        group_start = T.alloc_var("int32", init=0)
                        if lo > 0:
                            group_start = T.ceildiv(groups[lo - 1], alignment) * alignment
                        while lo < num_groups and group_start < (base + group_extent) * group_divisor:
                            end = groups[lo]
                            lower = T.max(0, T.min(group_extent, group_start // group_divisor - base))
                            upper = T.max(0, T.min(group_extent, T.ceildiv(end, group_divisor) - base))
                            if lower < upper:
                                group_ranges_ub[num_ranges, 0] = lower
                                group_ranges_ub[num_ranges, 1] = upper
                                num_ranges = num_ranges + 1
                            group_start = T.ceildiv(end, alignment) * alignment
                            lo = lo + 1
                    if major != Major.K or is_float or grouped:
                        with T.SimdVF():
                            full = S.pset(bits)
                            zero = S.vdup(0, raw_dtype)
                            if is_float:
                                validation = S.alloc_var("uint32")
                                validation = S.vld(validation_ub[0])
                            if clear_rows:
                                for segment in T.Unroll(2, explicit=True):
                                    begin = T.if_then_else(segment == 0, 0, upper)
                                    end = T.if_then_else(segment == 0, lower, actual_src_mn)
                                    for row in T.serial(begin, end):
                                        for col in T.Unroll(T.ceildiv(src_block_k, lanes), explicit=True):
                                            S.vsts(source_bits[row, col * lanes], zero, full, dist=f"NORM_B{bits}")
                                S.mem_bar("VST_VLD")
                            elif grouped and not scalar_k_mask:
                                for chunk in T.serial(T.ceildiv(group_extent, lanes)):
                                    coord = T.reinterpret(S.vci(chunk * lanes, f"int{bits}"), f"{raw_dtype}x{lanes}")
                                    valid_group = S.alloc_var("bool")
                                    valid_group = S.pnot(full, full)
                                    for group_idx in T.serial(num_ranges):
                                        lower = group_ranges_ub[group_idx, 0]
                                        upper = group_ranges_ub[group_idx, 1]
                                        after = S.vcmps(coord, T.cast(lower, raw_dtype), full, "ge")
                                        before = S.vcmps(coord, T.cast(upper, raw_dtype), full, "lt")
                                        valid_group = S.por(valid_group, S.pand(after, before, full), full)
                                    flags = S.vsel(S.vdup(T.cast((1 << bits) - 1, raw_dtype), raw_dtype), zero, valid_group)
                                    S.vsts(group_mask_ub[chunk * lanes], flags, full, dist=f"NORM_B{bits}")
                                S.mem_bar("VST_VLD")
                            if major == Major.K:
                                masks = S.alloc_local((T.ceildiv(src_block_k, lanes),), "bool")
                                for col in T.Unroll(T.ceildiv(src_block_k, lanes), explicit=True):
                                    masks[col] = S.vcmps(S.vci(col * lanes, f"int{bits}"), actual_src_k, full, "lt")
                                for row in T.serial(actual_src_mn):
                                    for col in T.Unroll(T.ceildiv(src_block_k, lanes), explicit=True):
                                        value = S.alloc_var(raw_dtype)
                                        value = S.vsel(S.vld(source_bits[row, col * lanes]), zero, masks[col])
                                        if m_grouped and not clear_rows:
                                            value = S.vand(value, S.vld(group_mask_ub[row], dist=f"BRC_B{bits}"), full)
                                        elif k_grouped:
                                            value = S.vand(value, S.vld(group_mask_ub[col * lanes]), full)
                                        if is_float:
                                            validation = S.vor(validation, value, full)
                                            S.vsts(packed_bytes[row, col * lanes], S.vshrs(value, 23, full), full, dist="PK4_B32")
                                        else:
                                            S.vsts(packed_ub[row, col * lanes], value, full, dist="NORM_B16")
                            else:
                                for pair in T.serial(actual_pairs):
                                    if k_grouped:
                                        col_flags = S.alloc_local((pack,), raw_dtype)
                                        for part in T.Unroll(pack, explicit=True):
                                            if scalar_k_mask:
                                                valid_col = T.cast(pair * pack + part >= lower, "int32") & T.cast(
                                                    pair * pack + part < upper, "int32"
                                                )
                                                col_flags[part] = S.vdup(T.cast(-valid_col, raw_dtype), raw_dtype)
                                            else:
                                                col_flags[part] = S.vld(group_mask_ub[pair * pack + part], dist=f"BRC_B{bits}")
                                    for chunk in T.serial(T.ceildiv(actual_src_mn, lanes)):
                                        coord = S.vci(chunk * lanes, f"int{bits}")
                                        row_mask = S.vcmps(coord, actual_src_mn, full, "lt")
                                        packed = S.alloc_var(raw_dtype)
                                        packed = zero
                                        for part in T.Unroll(pack, explicit=True):
                                            col = pair * pack + part
                                            col_mask = S.vcmps(
                                                S.vdup(T.cast(col, raw_dtype), raw_dtype), T.cast(actual_src_k, raw_dtype), full, "lt"
                                            )
                                            valid = S.pand(row_mask, col_mask, full)
                                            value = S.alloc_var(raw_dtype)
                                            value = S.vsel(S.vld(source_bits[col, chunk * lanes]), zero, valid)
                                            if m_grouped:
                                                value = S.vand(value, S.vld(group_mask_ub[chunk * lanes]), full)
                                            elif k_grouped:
                                                value = S.vand(value, col_flags[part], full)
                                            if is_float:
                                                validation = S.vor(validation, value, full)
                                                packed = S.vor(packed, S.vshls(S.vshrs(value, 23, full), part * 8, full), full)
                                            else:
                                                packed = value
                                        S.vsts(converted[pair, chunk * lanes], packed, full, dist="PK_B32" if is_float else "NORM_B16")
                            if is_float:
                                S.vsts(validation_ub[0], validation)
                    if major == Major.K:
                        with T.SimdVF():
                            transpose_mask = S.pset(16)
                            for row_block in T.serial(T.ceildiv(actual_src_mn, 16)):
                                _transpose_16rows(packed_ub, converted, row_block * 16, block_pairs, converted_stride, transpose_mask)
                    if gran_mn != 1:
                        with T.SimdVF():
                            full16 = S.pset(16)
                            regs = S.alloc_local((max(2, repeat_regs),), "uint16")
                            for pair in T.serial(actual_pairs):
                                if gran_mn >= 16:
                                    src_ptr = S.make_ubuf_ptr(T.access_ptr(converted[pair, 0], "r", extent=converted_stride), "uint16")
                                    for chunk in T.serial(T.ceildiv(actual_mn, repeat_regs * 128)):
                                        regs[0], src_ptr = S.vld(src_ptr, dist="E2B_B16", post_inc=repeat_regs * 128 // gran_mn)
                                        for level in T.Unroll(gran_shift - 4, explicit=True):
                                            for j in T.Unroll(max(1, repeat_regs // 2), explicit=True):
                                                if j < (1 << level):
                                                    idx = T.min(1 << level, max(1, repeat_regs // 2)) - 1 - j
                                                    regs[2 * idx], regs[2 * idx + 1] = S.vintlv(regs[idx], regs[idx])
                                        for j in T.Unroll(repeat_regs, explicit=True):
                                            S.vsts(output_bits[pair, (chunk * repeat_regs + j) * 128], regs[j], full16, dist="NORM_B16")
                                else:
                                    for chunk in T.serial(T.ceildiv(actual_mn, 128)):
                                        coord = T.reinterpret(S.vci(chunk * 128, "int16"), "uint16x128")
                                        offset = S.vshrs(coord, gran_shift, full16)
                                        repeated = S.vgather2(
                                            T.access_ptr(converted[pair, 0], "r", extent=converted_stride), offset, full16
                                        )
                                        S.vsts(output_bits[pair, chunk * 128], repeated, full16, dist="NORM_B16")
                    T.copy(
                        output_ub[:actual_pairs, :actual_mn],
                        output[batch, pair_idx : pair_idx + actual_pairs, mn_idx : mn_idx + actual_mn],
                        l2_cache_ctrl=config.l2_ctrl_store,
                    )
            if is_float:
                with T.SimdVF():
                    invalid = S.vand(S.vld(validation_ub[0]), S.vdup(T.uint32(0x807FFFFF), "uint32"))
                    S.vsts(validation_ub[0], S.vcmax(invalid), dist="ONEPT_B32")
                T.device_assert(validation_ub[0] == 0, no_stack_info=True)

    return main
