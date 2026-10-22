# Adapted from DeepSeek DeepSelect bfa4507d935f17ebfc3d0f00ff7d3c9a4d0e5c18.
# Copyright (c) 2025 DeepSeek. MIT license: see deepselect.LICENSE.
"""Radix selection expressed entirely with TileLang SIMD primitives."""

import tilelang.ascend.language as T
from tilelang.ascend.language import simd as S
from tvm.tirx.script.builder.ir import bind


def _mask(width, pattern="PAT_ALL"):
    return bind(S.pset(width, pattern))


def _u16(x):
    return T.reinterpret(x, "uint16x128")


def _bucket(h0, h1, rank):
    absent = S.vdup(65535, "uint16")
    lo = S.vsel(_u16(S.vci(0, "int16")), absent, S.vcmp(h0, rank, op="ge"))
    hi = S.vsel(_u16(S.vci(128, "int16")), absent, S.vcmp(h1, rank, op="ge"))
    return S.vdupv(S.vcmin(S.vmin(lo, hi)))


def _histogram(high):
    @T.macro
    def accumulate(raw, start, count, h, prefix):
        pos_flip = S.vdup(127 if high else 128, "uint8")
        neg_flip = S.vdup(0 if high else 255, "uint8")
        if not high:
            neg_prefix = S.vcmps(prefix, 128, op="lt")
            raw_prefix = S.vxor(prefix, S.vsel(neg_flip, pos_flip, neg_prefix))
        for chunk in T.serial(T.ceildiv(count, 256)):
            low, upper = S.vld2(raw[start * 2 + chunk * 512], "DINTLV_B8")
            active = S.update_mask(count - chunk * 256, 8)
            if high:
                nonnegative = S.vcmps(T.reinterpret(upper, "int8x256"), 0, op="ge")
                key = S.alloc_local((1,), "uint8")
                key[0] = upper
                key[0] = S.vxor(upper, pos_flip, nonnegative, mode="MODE_MERGING")
                S.chistv2(h[0], key[0], active, bin=0)
                S.chistv2(h[1], key[0], active, bin=1)
            else:
                key = S.alloc_local((1,), "uint8")
                key[0] = low
                key[0] = S.vxor(low, neg_flip, neg_prefix, mode="MODE_MERGING")
                selected = S.vcmp(upper, raw_prefix, active, op="eq")
                S.chistv2(h[0], key[0], selected, bin=0)
                S.chistv2(h[1], key[0], selected, bin=1)

    return accumulate


_high_histogram = _histogram(True)
_low_histogram = _histogram(False)


@T.macro
def _nan_count(values, start, count, state, state_offset, initialize):
    all_valid = S.alloc_var("bool")
    all_valid = _mask(16)
    full = _mask(16)
    for chunk in T.serial(T.ceildiv(count, 128)):
        active = S.update_mask(count - chunk * 128, 16)
        value = S.vld(values[start + chunk * 128])
        valid = S.vcmp(value, value, active, op="eq")
        all_valid = S.psel(S.psel(valid, full, active), all_valid, all_valid)
    missing = S.pnot(all_valid, full)
    total = S.vdupv(S.vcadd(S.vdup(1, "uint16", missing)))
    if initialize:
        S.vsts(state[state_offset + 8], total, dist="NORM_B32")
    else:
        S.vsts(state[state_offset + 8], S.vadd(total, S.vld(state[state_offset + 8], "BRC_B32")), dist="NORM_B32")


def make_simd(capacity, filtered):
    part = (capacity + 16384 + filtered + 256) * 2
    hist = part * 3 // 2
    high_hist = hist + 256
    threshold = high_hist + 256
    state_offset = (threshold * 2 + 32) // 4
    positions = (state_offset * 4 + 288) // 2
    page_map = positions + capacity
    inp = max(page_map * 2 + 2080, 131072)

    @T.macro
    def histogram(w, page, count, initialize, k):
        values = T.view(w, (126976,), "bfloat16")
        u16 = T.view(w, (126976,), "uint16")
        state = T.view(w, (63488,), "uint32")
        with T.SimdVF():
            h = S.alloc_local((2,), "uint16")
            if initialize:
                h[0] = S.vdup(0, "uint16")
                h[1] = S.vdup(0, "uint16")
            else:
                h[0] = S.vld(u16[high_hist])
                h[1] = S.vld(u16[high_hist + 128])
            _high_histogram(w, page * 2048, count, h, S.vdup(0, "uint8"))
            _nan_count(values, page * 2048, count, state, state_offset, initialize)
            S.vsts(u16[high_hist], S.vmins(h[0], k + 1), dist="NORM_B16")
            S.vsts(u16[high_hist + 128], S.vmins(h[1], k + 1), dist="NORM_B16")

    def select_factory(warmup, final):
        @T.macro
        def select(w, input_count, stage, virtual_base, k):
            values = T.view(w, (126976,), "bfloat16")
            u16 = T.view(w, (126976,), "uint16")
            u32 = T.view(w, (63488,), "uint32")
            state = T.view(w, (63488,), "uint32")
            with T.SimdVF():
                source_count = T.alloc_var("int32")
                source_count = input_count
                if not warmup:
                    source_count = T.cast(state[state_offset], "int32")
                    before = source_count
                    threshold_value = T.reinterpret(S.vld(u16[threshold], "BRC_B16"), "bfloat16x128")
                    stream = S.compact_store_begin(T.access_ptr(u16[part // 2 + before], "w", extent=16384))
                    input_ptr = S.make_ubuf_ptr(T.access_ptr(values[inp // 2 + stage * 16384], "r", extent=16384), "bfloat16")
                    filter_index = S.alloc_var("uint16")
                    filter_index = _u16(S.vci(T.cast(virtual_base, "int16"), "int16"))
                    for _chunk in T.serial(T.ceildiv(input_count, 128)):
                        value, input_ptr = S.vld(input_ptr, post_inc=128)
                        selected = S.pnot(S.vcmp(value, threshold_value, op="le"), _mask(16))
                        S.compact_store(stream, filter_index, selected)
                        filter_index = S.vadds(filter_index, 128)
                    S.compact_store_end(stream, u32[hist // 2])
                    S.mem_bar("VST_LD")
                    S.mem_bar("VST_VLD")
                    source_count = before + T.cast(u32[hist // 2] // 2, "int32")
                    added = source_count - before
                    full = _mask(16)
                    index_high = S.vdup(T.cast(virtual_base >> 16, "uint16"), "uint16")
                    gather_base = inp // 2 + stage * 16384 - T.cast(T.cast(virtual_base, "uint16"), "int32")
                    for chunk in T.serial(added // 128):
                        low = S.vld(u16[part // 2 + before + chunk * 128])
                        value = S.vgather2(values[gather_base], low, full)
                        S.vsts(u16[before + chunk * 128], _u16(value), full, "NORM_B16")
                        S.vsts(u16[part + before + chunk * 128], index_high, full, "NORM_B16")
                    if added % 128 != 0:
                        tail_start = before + added // 128 * 128
                        active = S.update_mask(added % 128, 16)
                        store_mask = S.update_mask(T.ceildiv(added % 128, 16) * 16, 16)
                        low = S.vld(u16[part // 2 + tail_start])
                        value = S.vgather2(values[gather_base], low, active)
                        padded = S.vsel(_u16(value), S.vdup(0xFF80, "uint16"), active)
                        S.vsts(u16[tail_start], padded, store_mask, "NORM_B16")
                        S.vsts(u16[part + tail_start], index_high, active, "NORM_B16")
                    source_count = T.ceildiv(source_count, 16) * 16
                if warmup or final or source_count > capacity + filtered:
                    S.mem_bar("VST_VLD")
                    if warmup:
                        k_vf = k
                    else:
                        k_vf = T.alloc_var("int32")
                        k_vf = T.cast(state[state_offset + 1], "int32")
                    h = S.alloc_local((2,), "uint16")
                    h[0] = S.vld(u16[high_hist])
                    h[1] = S.vld(u16[high_hist + 128])
                    if not warmup and source_count > T.ceildiv(k_vf, 16) * 16:
                        _high_histogram(w, T.ceildiv(k_vf, 16) * 16, source_count - T.ceildiv(k_vf, 16) * 16, h, S.vdup(0, "uint8"))
                        _nan_count(values, T.ceildiv(k_vf, 16) * 16, source_count - T.ceildiv(k_vf, 16) * 16, state, state_offset, False)
                        h[0] = S.vmins(h[0], k_vf + 1)
                        h[1] = S.vmins(h[1], k_vf + 1)
                        S.vsts(u16[high_hist], h[0], dist="NORM_B16")
                        S.vsts(u16[high_hist + 128], h[1], dist="NORM_B16")
                    bucket = _bucket(h[0], h[1], S.vdup(k_vf, "uint16"))
                    lane = S.vand(S.vadds(bucket, 65535), S.vdup(127, "uint16"))
                    chosen = S.vsel(h[0], h[1], S.vcmps(bucket, 128, op="le"))
                    previous = S.vsel(S.vdup(0, "uint16"), S.vselr(chosen, lane), S.vcmps(bucket, 0, op="eq"))
                    rank = S.vsub(S.vdup(k_vf, "uint16"), previous)
                    high_bucket = S.vsub(S.vdup(255, "uint16"), bucket)
                    prefix, unused = S.vdintlv(T.reinterpret(high_bucket, "uint8x256"), T.reinterpret(high_bucket, "uint8x256"))
                    h[0] = S.vdup(0, "uint16")
                    h[1] = S.vdup(0, "uint16")
                    _low_histogram(w, 0, source_count, h, prefix)
                    bucket_count = S.vdupv(h[1], pos="POS_HIGHEST")
                    low_bucket = _bucket(h[0], h[1], S.vadds(S.vsub(bucket_count, rank), 1))
                    lane = S.vand(low_bucket, S.vdup(127, "uint16"))
                    chosen = S.vsel(h[0], h[1], S.vcmps(low_bucket, 128, op="lt"))
                    pivot_count = S.vselr(chosen, lane)
                    greater_count = S.vadd(previous, S.vsub(bucket_count, pivot_count))
                    equal_limit = S.vsub(S.vdup(k_vf, "uint16"), greater_count)
                    key = S.vor(S.vshls(high_bucket, 8), low_bucket)
                    flip = S.vsel(S.vdup(0x8000, "uint16"), S.vdup(0xFFFF, "uint16"), S.vcmps(high_bucket, 128, op="ge"))
                    threshold_key = S.vxor(key, flip)
                    S.vsts(u16[threshold], threshold_key, _mask(16, "PAT_VL1"), "NORM_B16")
                    negative = S.vcmps(T.reinterpret(threshold_key, "int16x128"), 0, op="lt")
                    compare_flip = S.vsel(S.vdup(0x7FFF, "uint16"), S.vdup(0, "uint16"), negative)
                    compare_threshold = T.reinterpret(S.vxor(threshold_key, compare_flip), "int16x128")
                    equal_seen = S.alloc_var("uint16")
                    equal_seen = S.vdup(0, "uint16")
                    candidate_index = S.alloc_var("uint16")
                    candidate_index = _u16(S.vci(0, "int16"))
                    stream = S.compact_store_begin(T.access_ptr(u16[positions], "w", extent=capacity))
                    for chunk in T.serial(T.ceildiv(source_count, 128)):
                        active = S.update_mask(source_count - chunk * 128, 16)
                        bits = S.vld(u16[chunk * 128])
                        compare = T.reinterpret(S.vxor(bits, compare_flip, active), "int16x128")
                        greater = S.vcmp(compare, compare_threshold, active, op="gt")
                        equal = S.vcmp(compare, compare_threshold, active, op="eq")
                        prefix_count = _u16(S.vusqz(equal, "int16"))
                        inclusive = S.vsel(S.vadds(prefix_count, 1), prefix_count, equal)
                        equal_prefix = S.vadd(inclusive, equal_seen)
                        take_equal = S.vcmp(equal_limit, equal_prefix, equal, op="ge")
                        selected = S.por(greater, take_equal, active)
                        S.compact_store(stream, candidate_index, selected)
                        equal_seen = S.vdupv(equal_prefix, pos="POS_HIGHEST")
                        candidate_index = S.vadds(candidate_index, 128)
                    S.compact_store_end(stream)
                    S.mem_bar("VST_VLD")
                    for chunk in T.serial(T.ceildiv(k_vf, 128)):
                        active = S.update_mask(k_vf - chunk * 128, 16)
                        position = S.vld(u16[positions + chunk * 128])
                        value = S.vgather2(values[0], position, active)
                        if warmup:
                            low = position
                            high = S.vdup(0, "uint16")
                        else:
                            low = S.vgather2(u16[part // 2], position, active)
                            high = S.vgather2(u16[part], position, active)
                        S.vsts(u16[chunk * 128], _u16(value), active, "NORM_B16")
                        S.vsts(u16[part // 2 + chunk * 128], low, active, "NORM_B16")
                        S.vsts(u16[part + chunk * 128], high, active, "NORM_B16")
                    S.mem_bar("VST_VLD")
                    tail = S.vld(u16[k_vf // 16 * 16])
                    valid = S.update_mask(k_vf % 16, 16)
                    store_mask = S.update_mask(T.ceildiv(k_vf, 16) * 16 - k_vf // 16 * 16, 16)
                    S.vsts(u16[k_vf // 16 * 16], S.vsel(tail, S.vdup(65535, "uint16"), valid), store_mask, "NORM_B16")
                    source_count = T.ceildiv(k_vf, 16) * 16
                if warmup:
                    state[state_offset + 1] = T.cast(k, "uint32")
                if not warmup:
                    state[state_offset] = T.cast(source_count, "uint32")

        return select

    @T.macro
    def short_row(w, n, k, return_value, index_fill, value_fill):
        values = T.view(w, (126976,), "bfloat16")
        u16 = T.view(w, (126976,), "uint16")
        idx = T.view(w, (63488,), "int32")
        state = T.view(w, (63488,), "uint32")
        with T.SimdVF():
            _nan_count(values, inp // 2, n, state, state_offset, True)
            if return_value:
                for chunk in T.serial(T.ceildiv(k, 128)):
                    active = S.update_mask(k - chunk * 128, 16)
                    valid = S.update_mask(T.max(0, T.min(128, n - chunk * 128)), 16)
                    value = S.vld(u16[inp // 2 + chunk * 128])
                    S.vsts(u16[chunk * 128], S.vsel(value, S.vdup(value_fill, "uint16"), valid), active, "NORM_B16")
            for chunk in T.serial(T.ceildiv(k, 64)):
                active = S.update_mask(k - chunk * 64, 32)
                valid = S.update_mask(T.max(0, T.min(64, n - chunk * 64)), 32)
                index = S.vci(chunk * 64, "int32")
                S.vsts(idx[part // 4 + chunk * 64], S.vsel(index, S.vdup(index_fill, "int32"), valid), active, "NORM_B32")

    @T.macro
    def output(w, n, offset, k):
        u16 = T.view(w, (126976,), "uint16")
        idx = T.view(w, (63488,), "int32")
        state = T.view(w, (63488,), "uint32")
        with T.SimdVF():
            if state[state_offset + 8] != 0:
                for chunk in T.serial(T.ceildiv(k, 64)):
                    active = S.update_mask(k - chunk * 64, 32)
                    S.vsts(idx[inp // 4 + chunk * 64], S.vdup(0x3F3F3F3F, "int32"), active, "NORM_B32")
            elif n > k:
                for chunk in T.serial(T.ceildiv(k, 128)):
                    active = S.update_mask(k - chunk * 128, 16)
                    low = S.vld(u16[part // 2 + chunk * 128])
                    high = S.vld(u16[part + chunk * 128])
                    virtual_page = S.vor(S.vshrs(low, 11), S.vshls(high, 5))
                    physical = S.vgather2(u16[page_map], virtual_page, active)
                    local = S.vand(low, S.vdup(2047, "uint16"))
                    off0, off1 = S.vintlv(local, S.vdup(0, "uint16"))
                    base0, base1 = S.vintlv(physical, S.vdup(0, "uint16"))
                    index0 = S.vadd(S.vshls(T.reinterpret(base0, "uint32x64"), 7), T.reinterpret(off0, "uint32x64"))
                    index1 = S.vadd(S.vshls(T.reinterpret(base1, "uint32x64"), 7), T.reinterpret(off1, "uint32x64"))
                    mask0, mask1 = S.pintlv(active, _mask(16, "PAT_ALLF"), width=16)
                    S.vsts(idx[inp // 4 + chunk * 128], S.vadds(T.reinterpret(index0, "int32x64"), offset), mask0, "NORM_B32")
                    S.vsts(idx[inp // 4 + chunk * 128 + 64], S.vadds(T.reinterpret(index1, "int32x64"), offset), mask1, "NORM_B32")
            else:
                for chunk in T.serial(T.ceildiv(k, 64)):
                    active = S.update_mask(k - chunk * 64, 32)
                    valid = S.update_mask(T.max(0, T.min(64, n - chunk * 64)), 32)
                    index = S.vld(idx[part // 4 + chunk * 64])
                    result = S.vsel(S.vadds(index, offset), index, valid)
                    S.vsts(idx[inp // 4 + chunk * 64], result, active, "NORM_B32")

    return histogram, select_factory, short_row, output


def make_index_sort(capacity, inp, part):
    output_key = inp + 4 * capacity
    output_perm = inp + 8 * capacity
    scratch = inp + 12 * capacity
    assert scratch + 512 + 11 * capacity <= 253952

    @T.macro
    def global_offsets(w, source, byte, k):
        aligned = T.ceildiv(k, 32) * 32
        digit = scratch + 512
        saved_digit = digit + 3 * aligned
        u32 = T.view(w, (63488,), "uint32")
        u16 = T.view(w, (126976,), "uint16")
        with T.SimdVF():
            h = S.alloc_local((4,), "uint16")
            for i in T.unroll(4):
                h[i] = S.vdup(0, "uint16")
            for block in T.serial(T.ceildiv(k, 256)):
                active = S.update_mask(T.min(256, k - block * 256), 8)
                p0 = S.vshrs(S.vld(u32[source // 4 + block * 256]), byte * 8)
                p1 = S.vshrs(S.vld(u32[source // 4 + block * 256 + 64]), byte * 8)
                p2 = S.vshrs(S.vld(u32[source // 4 + block * 256 + 128]), byte * 8)
                p3 = S.vshrs(S.vld(u32[source // 4 + block * 256 + 192]), byte * 8)
                half0, unused0 = S.vdintlv(T.reinterpret(p0, "uint16x128"), T.reinterpret(p1, "uint16x128"))
                half1, unused1 = S.vdintlv(T.reinterpret(p2, "uint16x128"), T.reinterpret(p3, "uint16x128"))
                key, unused2 = S.vdintlv(T.reinterpret(half0, "uint8x256"), T.reinterpret(half1, "uint8x256"))
                S.vsts(w[digit + block * 256], key, active, "NORM_B8")
                S.vsts(w[saved_digit + block * 256], key, active, "NORM_B8")
                S.dhistv2(h[0], key, active, bin=0)
                S.dhistv2(h[1], key, active, bin=1)
                S.chistv2(h[2], key, active, bin=0)
                S.chistv2(h[3], key, active, bin=1)
            S.vsts(u16[scratch // 2], S.vsub(h[2], h[0]), dist="NORM_B16")
            S.vsts(u16[scratch // 2 + 128], S.vsub(h[3], h[1]), dist="NORM_B16")

    @T.macro
    def sort_digit_blocks(w, k):
        aligned = T.ceildiv(k, 32) * 32
        digit = scratch + 512
        local16 = digit + aligned
        local32 = digit + 3 * aligned
        saved_digit = local32
        local8 = saved_digit + aligned
        u16 = T.view(w, (126976,), "uint16")
        with T.SimdVF():
            ramp = T.reinterpret(S.vci(0, "int8"), "uint8x256")
            for block in T.serial(T.ceildiv(k, 256)):
                active = S.update_mask(T.min(256, k - block * 256), 8)
                S.vsts(w[local8 + block * 256], ramp, active, "NORM_B8")
            S.mem_bar("VST_VLD")
            for pair in T.serial(4):
                stream = S.compact_store_begin(T.access_ptr(w[local8], "w", extent=aligned))
                shift = S.vdup(T.cast(pair * 2, "int8"), "int8")
                for block in T.serial(T.ceildiv(k, 256)):
                    active = S.update_mask(T.min(256, k - block * 256), 8)
                    key = S.vld(w[digit + block * 256])
                    order = S.vld(w[local8 + block * 256])
                    bits = S.vand(S.vshr(key, shift), S.vdup(3, "uint8"), active)
                    for bucket in T.unroll(4):
                        chosen = S.vcmps(bits, bucket, active, op="eq")
                        S.compact_store(stream, order, chosen)
                S.compact_store_end(stream)
                S.mem_bar("VST_VLD")
                # Match the reference post-update reload after in-place compaction.
                read_order = S.make_ubuf_ptr(T.access_ptr(w[local8], "r", extent=aligned), "uint8")
                for block in T.serial(T.ceildiv(k, 256)):
                    active = S.update_mask(T.min(256, k - block * 256), 8)
                    original = S.vld(w[saved_digit + block * 256])
                    order, read_order = S.vld(read_order, post_inc=256)
                    S.vsts(w[digit + block * 256], S.vselr(original, order), active, "NORM_B8")
                S.mem_bar("VST_VLD")
            for block in T.serial(T.ceildiv(k, 256)):
                order = S.vld(w[local8 + block * 256])
                lo, hi = S.vintlv(order, S.vdup(0, "uint8"))
                mask0 = S.update_mask(T.min(128, k - block * 256), 16)
                mask1 = S.update_mask(T.max(0, T.min(128, k - block * 256 - 128)), 16)
                S.vsts(u16[local16 // 2 + block * 256], S.vadds(_u16(lo), T.cast(block * 256, "uint16")), mask0, "NORM_B16")
                S.vsts(u16[local16 // 2 + block * 256 + 128], S.vadds(_u16(hi), T.cast(block * 256, "uint16")), mask1, "NORM_B16")

    @T.macro
    def scatter_permutation(w, destination, k):
        digit = scratch + 512
        local16 = digit + T.ceildiv(k, 32) * 32
        u16 = T.view(w, (126976,), "uint16")
        u32 = T.view(w, (63488,), "uint32")
        with T.SimdVF():
            offsets = S.alloc_local((2,), "uint16")
            offsets[0] = S.vld(u16[scratch // 2])
            offsets[1] = S.vld(u16[scratch // 2 + 128])
            h = S.alloc_local((4,), "uint16")
            zero = S.vdup(0, "uint8")
            lanes0 = _u16(S.vci(0, "int16"))
            lanes1 = _u16(S.vci(128, "int16"))
            for block in T.serial(T.ceildiv(k, 256)):
                active = S.update_mask(T.min(256, k - block * 256), 8)
                key = S.vld(w[digit + block * 256])
                index0 = S.vld(u16[local16 // 2 + block * 256])
                index1 = S.vld(u16[local16 // 2 + block * 256 + 128])
                offset_low, offset_high = S.vdintlv(T.reinterpret(offsets[0], "uint8x256"), T.reinterpret(offsets[1], "uint8x256"))
                for j in T.unroll(4):
                    h[j] = S.vdup(0, "uint16")
                S.dhistv2(h[0], key, active, bin=0)
                S.dhistv2(h[1], key, active, bin=1)
                S.chistv2(h[2], key, active, bin=0)
                S.chistv2(h[3], key, active, bin=1)
                prefix0 = S.vsub(h[2], h[0])
                prefix1 = S.vsub(h[3], h[1])
                prefix_low, unused = S.vdintlv(T.reinterpret(prefix0, "uint8x256"), T.reinterpret(prefix1, "uint8x256"))
                base_low = S.vselr(offset_low, key)
                base_high = S.vselr(offset_high, key)
                base0, base1 = S.vintlv(base_low, base_high)
                local_prefix = S.vselr(prefix_low, key)
                local0, local1 = S.vintlv(local_prefix, zero)
                rank0 = S.vsub(S.vadd(_u16(base0), lanes0), _u16(local0))
                rank1 = S.vsub(S.vadd(_u16(base1), lanes1), _u16(local1))
                rank00, rank01 = S.vintlv(rank0, _u16(zero))
                rank10, rank11 = S.vintlv(rank1, _u16(zero))
                order00, order01 = S.vintlv(index0, _u16(zero))
                order10, order11 = S.vintlv(index1, _u16(zero))
                m0 = S.update_mask(T.min(64, k - block * 256), 32)
                m1 = S.update_mask(T.max(0, T.min(64, k - block * 256 - 64)), 32)
                m2 = S.update_mask(T.max(0, T.min(64, k - block * 256 - 128)), 32)
                m3 = S.update_mask(T.max(0, T.min(64, k - block * 256 - 192)), 32)
                S.vscatter(T.reinterpret(order00, "uint32x64"), u32[destination // 4], T.reinterpret(rank00, "uint32x64"), m0)
                S.vscatter(T.reinterpret(order01, "uint32x64"), u32[destination // 4], T.reinterpret(rank01, "uint32x64"), m1)
                S.vscatter(T.reinterpret(order10, "uint32x64"), u32[destination // 4], T.reinterpret(rank10, "uint32x64"), m2)
                S.vscatter(T.reinterpret(order11, "uint32x64"), u32[destination // 4], T.reinterpret(rank11, "uint32x64"), m3)
                offsets[0] = S.vadd(offsets[0], h[0])
                offsets[1] = S.vadd(offsets[1], h[1])

    def reorder_factory(first):
        @T.macro
        def reorder(w, source, destination, order, previous_perm, next_perm, k):
            u32 = T.view(w, (63488,), "uint32")
            with T.SimdVF():
                for block in T.serial(T.ceildiv(k, 64)):
                    active = S.update_mask(T.min(64, k - block * 64), 32)
                    position = S.vld(u32[order // 4 + block * 64])
                    value = S.vgather2(u32[source // 4], position, active)
                    S.vsts(u32[destination // 4 + block * 64], value, active, "NORM_B32")
                    if not first:
                        permutation = S.vgather2(u32[previous_perm // 4], position, active)
                        S.vsts(u32[next_perm // 4 + block * 64], permutation, active, "NORM_B32")

        return reorder

    first_reorder = reorder_factory(True)
    reorder = reorder_factory(False)

    @T.macro
    def sort(w, k, return_value):
        aligned = T.ceildiv(k, 32) * 32
        local32 = scratch + 512 + 3 * aligned
        alternate_perm = local32 + 4 * aligned
        u32 = T.view(w, (63488,), "uint32")
        values = T.view(w, (126976,), "bfloat16")
        with T.SimdVF():
            for block in T.serial(T.ceildiv(k, 64)):
                active = S.update_mask(T.min(64, k - block * 64), 32)
                key = S.vld(u32[inp // 4 + block * 64])
                S.vsts(u32[output_key // 4 + block * 64], S.vxor(key, S.vdup(0x80000000, "uint32")), active, "NORM_B32")
        for byte in T.unroll(4):
            source = output_key if byte % 2 == 0 else inp
            destination = inp if byte % 2 == 0 else output_key
            previous_perm = output_perm if byte % 2 == 0 else alternate_perm
            next_perm = alternate_perm if byte % 2 == 0 else output_perm
            global_offsets(w, source, byte, k)
            sort_digit_blocks(w, k)
            if byte == 0:
                scatter_permutation(w, alternate_perm, k)
                first_reorder(w, source, destination, alternate_perm, previous_perm, next_perm, k)
            else:
                scatter_permutation(w, local32, k)
                reorder(w, source, destination, local32, previous_perm, next_perm, k)
        with T.SimdVF():
            for block in T.serial(T.ceildiv(k, 64)):
                active = S.update_mask(T.min(64, k - block * 64), 32)
                key = S.vld(u32[output_key // 4 + block * 64])
                S.vsts(u32[output_key // 4 + block * 64], S.vxor(key, S.vdup(0x80000000, "uint32")), active, "NORM_B32")
        if return_value:
            with T.SimdVF():
                for block in T.serial(T.ceildiv(k, 64)):
                    active = S.update_mask(T.min(64, k - block * 64), 16)
                    position = S.vpack(S.vld(u32[output_perm // 4 + block * 64]))
                    value = S.vgather2(values[0], position, active)
                    S.vsts(values[part + block * 64], value, active, "NORM_B16")

    return sort
