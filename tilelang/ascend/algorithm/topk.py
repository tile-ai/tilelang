"""Parameterized DeepSelect macro for Huawei Ascend with explicit pipeline scheduling."""

import tilelang.ascend.language as T
from ._topk_simd import make_simd, make_index_sort


def _policy(k):
    return (512, 2, 32832, 4096, 1536) if k <= 512 else ((1024, 3, 36800, 3072, 2560) if k <= 1024 else (4096, 4, 10432, 2560, 3072))


def _make_body(capacity, warm, filtered, sorted_index, abort_when_nan_found):
    part = (capacity + 16384 + filtered + 256) * 2
    state = part * 3 + 1056
    page_map = state + 288 + capacity * 2
    inp = max(page_map + 2080, 131072)
    histogram, select_factory, short_row, output = make_simd(capacity, filtered)
    warm_final = select_factory(True, True)
    warm_continue = select_factory(True, False)
    block_final = select_factory(False, True)
    block_continue = select_factory(False, False)
    sort = make_index_sort(capacity, inp, part)

    @T.macro
    def page_info(n, pages, size, step, logical, info):
        if pages <= warm:
            info[0] = logical * 2048
            info[1] = T.min(2048, n - info[0])
        else:
            last = T.ceildiv(n - 2048, 128) * 128
            if logical == warm - 1:
                info[0] = last
                info[1] = n - last
            elif logical == pages - 1:
                info[0] = (pages - 2) * 2048
                info[1] = last - info[0]
            else:
                info[0] = info[2] * 2048
                next_page = T.cast(info[2], "uint32") + step
                info[2] = T.cast(
                    T.if_then_else(next_page >= T.cast(size, "uint32"), next_page - T.cast(size, "uint32"), next_page), "int32"
                )
                info[1] = 2048

    @T.macro
    def body(X, I, V, w, row, valid_len, offset, k, return_value, index_fill, value_fill_bits):
        values = T.view(w, (126976,), "bfloat16")
        u16 = T.view(w, (126976,), "uint16")
        u32 = T.view(w, (63488,), "uint32")
        indices = T.view(w, (63488,), "int32")
        for stage in T.unroll(3):
            T.ascend_set_flag("V_MTE2", stage)
        if valid_len <= k:
            if valid_len != 0:
                T.ascend_wait_flag("V_MTE2", 0)
                T.copy(X[row, 0:valid_len], values[inp // 2 : inp // 2 + valid_len], l2_cache_ctrl="NOTALLOC_CLEAN")
                T.ascend_set_flag("MTE2_V", 0)
                T.ascend_wait_flag("MTE2_V", 0)
            short_row(w, valid_len, k, return_value, index_fill, value_fill_bits)
            if valid_len != 0:
                T.ascend_set_flag("V_MTE2", 0)
        else:
            row_input = T.make_tensor(T.address_of(X[row, 0]), (X.shape[1],), "bfloat16")
            seed = T.alloc_var("uint64")
            seed = T.uint64(0)
            if X.shape[1] >= 4:
                raw_input = T.StridedTensor((X.shape[0], T.ceildiv(X.shape[1], 4)), (X.strides[0] // 4, 1), "uint64", data=X.data)
                seed = raw_input[row, 0] ^ T.cast(valid_len, "uint64")
            info = T.alloc_local((3,), "int32")
            pages = (valid_len + 2047) // 2048
            size = T.max(pages - 2, 1)
            seed0 = T.cast(seed & T.uint64(65535), "uint32")
            seed1 = T.cast((seed >> 16) & T.uint64(65535), "uint32") % T.cast(size, "uint32")
            first = seed0 % T.cast(size, "uint32")
            reflected = T.if_then_else(first <= seed1, seed1 - first, T.cast(size, "uint32") + seed1 - first)
            info[2] = T.cast((reflected * T.uint32(65521) + seed1) % T.cast(size, "uint32"), "int32")
            step = T.alloc_var("uint32")
            step = T.cast(size, "uint32") - T.uint32(4294049777) % T.cast(size, "uint32")
            warm_pages = T.min(warm, pages)
            blocks = (pages - warm_pages + 7) // 8
            count = T.alloc_var("int32")
            count = 0
            for page in T.serial(warm_pages):
                page_info(valid_len, pages, size, step, page, info)
                start = info[0]
                length = info[1]
                # The page permutation returns a valid, at-most-2048-element slice.
                T.assume(start >= 0 and length >= 0 and length <= 2048)
                T.assume(start + length <= X.shape[1])
                T.copy(row_input[start : start + length], values[page * 2048 : page * 2048 + length], l2_cache_ctrl="NOTALLOC_CLEAN")
                T.ascend_set_flag("MTE2_V", page)
                T.ascend_wait_flag("MTE2_V", page)
                histogram(w, page, length, page == 0, k)
                u16[page_map // 2 + page] = T.cast(start // 128, "uint16")
                count += length
            if blocks == 0:
                warm_final(w, count, 0, 0, k)
            else:
                warm_continue(w, count, 0, 0, k)
            u32[state // 4] = T.uint32((k + 15) // 16 * 16)
            for batch in T.serial(blocks):
                stage = batch % 3
                T.ascend_wait_flag("V_MTE2", stage)
                count = 0
                for page in T.serial(8):
                    logical = warm_pages + batch * 8 + page
                    if logical < pages:
                        page_info(valid_len, pages, size, step, logical, info)
                        start = info[0]
                        length = info[1]
                        dst = inp // 2 + stage * 16384 + page * 2048
                        T.assume(start >= 0 and length >= 0 and length <= 2048)
                        T.assume(start + length <= X.shape[1])
                        T.copy(row_input[start : start + length], values[dst : dst + length], l2_cache_ctrl="NOTALLOC_CLEAN")
                        u16[page_map // 2 + (batch + 1) * 8 + page] = T.cast(start // 128, "uint16")
                        count += length
                T.ascend_set_flag("MTE2_V", stage)
                T.ascend_wait_flag("MTE2_V", stage)
                if batch + 1 == blocks:
                    block_final(w, count, stage, (batch + 1) * 16384, k)
                else:
                    block_continue(w, count, stage, (batch + 1) * 16384, k)
                T.ascend_set_flag("V_MTE2", stage)
        output(w, valid_len, offset, k)
        did_sort = sorted_index and k > 1 and valid_len > k and T.ceildiv(valid_len, 2048) > warm
        if did_sort:
            sort(w, k, return_value)
        T.ascend_set_flag("V_MTE3", 0)
        T.ascend_wait_flag("V_MTE3", 0)
        if did_sort:
            T.copy(indices[inp // 4 + capacity : inp // 4 + capacity + k], I[row, 0:k], l2_cache_ctrl="NORMAL_FV")
            if return_value:
                T.copy(values[part : part + k], V[row, 0:k], l2_cache_ctrl="NORMAL_FV")
        else:
            T.copy(indices[inp // 4 : inp // 4 + k], I[row, 0:k], l2_cache_ctrl="NORMAL_FV")
            if return_value:
                T.copy(values[0:k], V[row, 0:k], l2_cache_ctrl="NORMAL_FV")
        T.ascend_set_flag("MTE3_V", 0)
        T.ascend_wait_flag("MTE3_V", 0)
        for stage in T.unroll(3):
            T.ascend_wait_flag("V_MTE2", stage)
        if abort_when_nan_found:
            T.ascend_set_flag("V_S", 0)
            T.ascend_wait_flag("V_S", 0)
            T.device_assert(u32[state // 4 + 8] == 0, "TopK input contains NaN")

    return body


def make_topk_macro(capacity, *, sorted_index=False, abort_when_nan_found=True):
    """Create a runtime-parameterized macro for a 512/1024/4096 capacity tier.

    Owns 248 KiB UB and events 0..3; automatic scheduling must be disabled.
    Callers validate runtime shapes, aligned strides, K, and optional buffers.
    """
    if capacity not in (512, 1024, 4096):
        raise ValueError("TopK capacity must be 512, 1024, or 4096")
    _, warm, pivot, lower, upper = _policy(capacity)
    low = _make_body(capacity, warm, lower, sorted_index, abort_when_nan_found)
    high = _make_body(capacity, warm, upper, sorted_index, abort_when_nan_found)

    @T.macro
    def topk(X, I, V, end, offsets, row, n, k, return_value, has_end, has_offset, index_fill, value_fill_bits):
        w = T.alloc_shared((253952,), "uint8")
        valid = T.alloc_var("int32")
        offset = T.alloc_var("int32")
        valid = n
        offset = 0
        if has_end:
            valid = end[row]
        if has_offset:
            offset = offsets[row]
        if valid <= pivot:
            low(X, I, V, w, row, valid, offset, k, return_value, index_fill, value_fill_bits)
        else:
            high(X, I, V, w, row, valid, offset, k, return_value, index_fill, value_fill_bits)

    return topk
