#pragma once

#include "c_api/asc_simd.h"
#include <stdint.h>
#include <type_traits>
#include <version/asc_devkit_version.h>

// Stable names used by Ascend codegen. asc_simd.h selects the SDK APIs
// available for the compilation target.
namespace tl {
namespace cann {

enum class hf32_round_mode { NEAREST_AWAY, NEAREST_EVEN };

#if defined(__NPU_ARCH__)

#define TL_CANN_9_1                                                            \
  (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR == 1 && ASC_DEVKIT_PATCH == 0)

// HF32, atomic modes and scalar loads use the API names for this SDK.
#if TL_CANN_9_1
__aicore__ inline void disable_hf32() { ::asc_set_fp32_mode(); }

__aicore__ inline void set_hf32_round_mode(hf32_round_mode mode) {
  // The old boolean is a CTRL bit, not the newer enum's numeric value:
  // AWAY sets bit 47; EVEN clears it. Both APIs preserve other CTRL bits.
  ::asc_enable_hf32_trans(mode == hf32_round_mode::NEAREST_AWAY ? 1u : 0u);
}

__aicore__ inline void set_atomic_add_int32() { ::asc_set_atomic_add_int(); }
__aicore__ inline void set_atomic_max_int32() { ::asc_set_atomic_max_int(); }
__aicore__ inline void set_atomic_min_int32() { ::asc_set_atomic_min_int(); }

using load_l2_cache_mode = uint8_t;
using store_l2_cache_mode = uint8_t;
using unit_flag_mode = uint8_t;
using quant_mode = uint64_t;
using relu_pre_mode = uint8_t;
using dual_dst_mode = uint8_t;

template <typename T> __aicore__ inline T load_dev(__gm__ T *ptr) {
  return ld_dev(ptr, 0);
}
#else
__aicore__ inline void disable_hf32() { ::asc_disable_hf32(); }

__aicore__ inline void set_hf32_round_mode(hf32_round_mode mode) {
  ::asc_set_hf32_round_mode(mode == hf32_round_mode::NEAREST_AWAY
                                ? ::asc_hf32_round_mode::NEAREST_AWAY
                                : ::asc_hf32_round_mode::NEAREST_EVEN);
}

__aicore__ inline void set_atomic_add_int32() { ::asc_set_atomic_add_int32(); }
__aicore__ inline void set_atomic_max_int32() { ::asc_set_atomic_max_int32(); }
__aicore__ inline void set_atomic_min_int32() { ::asc_set_atomic_min_int32(); }

using load_l2_cache_mode = ::asc_load_l2_cache_mode;
using store_l2_cache_mode = ::asc_store_l2_cache_mode;
using unit_flag_mode = ::asc_unit_flag_mode;
using quant_mode = ::asc_quant_mode;
using relu_pre_mode = ::asc_relu_pre_mode;
using dual_dst_mode = ::asc_dual_dst_mode;

template <typename T> __aicore__ inline T load_dev(__gm__ T *ptr) {
  return ::asc_load_dev(ptr);
}
#endif

// The 9.1 dav-3510 DMA ABI uses packed controls and a longer L0C store
// signature.
#if TL_CANN_9_1 && __NPU_ARCH__ == 3510
__aicore__ inline void set_gm2l1_nz_para(uint16_t batches, uint16_t n_stride,
                                         uint16_t c0_stride,
                                         uint16_t matrix_stride) {
  uint64_t config = uint64_t(batches) | (uint64_t(n_stride) << 16) |
                    (uint64_t(c0_stride) << 32) |
                    (uint64_t(matrix_stride) << 48);
  ::asc_set_gm2l1_nz_para(config);
}

template <typename... Args>
__aicore__ inline void set_l0c_copy_nz_para(Args... args) {
  ::asc_set_l0c2gm_nz2nd(args...);
}

template <typename D, typename S>
__aicore__ inline void
copy_l0c2gm(__gm__ D *dst, __cc__ S *src, uint16_t n, uint16_t m,
            uint32_t dst_stride, uint16_t src_stride, store_l2_cache_mode cache,
            unit_flag_mode unit, quant_mode quant, relu_pre_mode relu,
            bool split, bool nz2nd, bool nz2dn, bool clip) {
  ::asc_copy_l0c2gm(
      dst, src, n, m, dst_stride, src_stride, cache, clip, unit, quant, relu,
      split, nz2nd, /*quant_post=*/0, /*relu_post=*/0, /*clip_relu_post=*/false,
      /*eltwise_op=*/0, /*eltwise_antq_en=*/false, /*c0_pad_en=*/false,
      /*broadcast_en=*/false, nz2dn);
}

template <typename D, typename S>
__aicore__ inline void
copy_l0c2ub(__ubuf__ D *dst, __cc__ S *src, uint16_t n, uint16_t m,
            uint32_t dst_stride, uint16_t src_stride, int8_t subblock,
            dual_dst_mode dual, unit_flag_mode unit, quant_mode quant,
            relu_pre_mode relu, bool split, bool nz2nd, bool nz2dn, bool clip) {
  ::asc_copy_l0c2ub(
      dst, src, n, m, dst_stride, src_stride, dual, bool(subblock), clip, unit,
      quant, relu, split, nz2nd, /*quant_post=*/0, /*relu_post=*/0,
      /*clip_relu_post=*/false, /*eltwise_op=*/0, /*eltwise_antq_en=*/false,
      /*c0_pad_en=*/false, /*broadcast_en=*/false, nz2dn);
}
#else
// Resolve the native overload when the operation is instantiated. A target
// may expose only a subset of these DMA operations.
template <typename... Args>
__aicore__ inline void set_gm2l1_nz_para(Args... args) {
  asc_set_gm2l1_nz_para(args...);
}
template <typename... Args>
__aicore__ inline void set_l0c_copy_nz_para(Args... args) {
  asc_set_l0c_copy_nz_para(args...);
}
template <typename... Args> __aicore__ inline void copy_l0c2gm(Args... args) {
  asc_copy_l0c2gm(args...);
}
template <typename... Args> __aicore__ inline void copy_l0c2ub(Args... args) {
  asc_copy_l0c2ub(args...);
}
#endif

// The 9.1 strided overloads narrow the burst count to uint16_t. Use the v2
// instruction when available to keep the uint32_t count. Other overloads
// forward to the SDK.
#if TL_CANN_9_1 && __has_builtin(__builtin_cce_copy_gm_to_ubuf_align_v2)
template <typename T>
__aicore__ inline void
copy_gm2ub_align_wide(__ubuf__ T *dst, __gm__ T *src, uint32_t count,
                      uint32_t length, uint8_t left, uint8_t right, bool pad,
                      load_l2_cache_mode cache, uint64_t src_stride,
                      uint32_t dst_stride) {
  if ASC_IS_AIV {
    copy_gm_to_ubuf_align_v2(dst, src, 0, count, length, left, right, pad,
                             cache, src_stride, dst_stride);
  }
}
#endif

template <typename... Args>
__aicore__ inline void copy_gm2ub_align(Args... args) {
  if constexpr (requires { copy_gm2ub_align_wide(args...); }) {
    copy_gm2ub_align_wide(args...);
  } else {
    asc_copy_gm2ub_align(args...);
  }
}

#if TL_CANN_9_1 && __has_builtin(__builtin_cce_copy_ubuf_to_gm_align_v2)
template <typename T>
__aicore__ inline void
copy_ub2gm_align_wide(__gm__ T *dst, __ubuf__ T *src, uint32_t count,
                      uint32_t length, store_l2_cache_mode cache,
                      uint64_t dst_stride, uint32_t src_stride) {
  if ASC_IS_AIV {
    copy_ubuf_to_gm_align_v2((__gm__ void *)dst, (__ubuf__ void *)src, 0, count,
                             length, cache, dst_stride, src_stride);
  }
}
#endif

template <typename... Args>
__aicore__ inline void copy_ub2gm_align(Args... args) {
  if constexpr (requires { copy_ub2gm_align_wide(args...); }) {
    copy_ub2gm_align_wide(args...);
  } else {
    asc_copy_ub2gm_align(args...);
  }
}

// asc_simd.h includes reg_vector.h in 9.1 and asc_reg_vector.h in newer
// SDKs when the target exposes register-vector operations.
#if defined(INCLUDE_C_API_REG_COMPUTE_REG_VECTOR_H) ||                         \
    defined(INCLUDE_C_API_ASC_REG_VECTOR_H)
template <typename V>
__simd_callee__ inline void div(V &dst, V lhs, V rhs, vector_bool mask) {
#if TL_CANN_9_1
  // CANN 9.1 lacks the integer asc_div overloads.
  if constexpr (std::is_same_v<V, vector_s16> ||
                std::is_same_v<V, vector_u16> ||
                std::is_same_v<V, vector_s32> ||
                std::is_same_v<V, vector_u32>) {
    ::vdiv(dst, lhs, rhs, mask, MODE_ZEROING);
  } else {
    ::asc_div(dst, lhs, rhs, mask);
  }
#else
  ::asc_div(dst, lhs, rhs, mask);
#endif
  // Unsigned division can leave quotient estimates in inactive lanes.
  // A masked add of zero preserves the quotient and clears those lanes.
  if constexpr (std::is_same_v<V, vector_u16>) {
    ::vadds(dst, dst, uint16_t(0), mask, MODE_ZEROING);
  } else if constexpr (std::is_same_v<V, vector_u32>) {
    ::vadds(dst, dst, uint32_t(0), mask, MODE_ZEROING);
  }
}

#if TL_CANN_9_1
template <typename V, typename S>
__simd_callee__ inline void min_scalar(V &dst, V src, S scalar,
                                       vector_bool mask) {
  if constexpr (std::is_same_v<V, vector_bf16>) {
    ::vmins(dst, src, scalar, mask, MODE_ZEROING);
  } else {
    ::asc_min_scalar(dst, src, scalar, mask);
  }
}

template <typename V>
__simd_callee__ inline void ne(vector_bool &dst, V lhs, V rhs,
                               vector_bool mask) {
  if constexpr (std::is_same_v<V, vector_bf16>) {
    ::vcmp_ne(dst, lhs, rhs, mask);
  } else {
    ::asc_ne(dst, lhs, rhs, mask);
  }
}

template <typename V>
__simd_callee__ inline void duplicate(V &dst, V src, vector_bool mask) {
  if constexpr (std::is_same_v<V, vector_f8e8m0>) {
    ::vdup(dst, src, mask, POS_LOWEST, MODE_ZEROING);
  } else {
    ::asc_duplicate(dst, src, mask);
  }
}

template <typename V>
__simd_callee__ inline void duplicate_highest(V &dst, V src, vector_bool mask) {
  ::vdup(dst, src, mask, POS_HIGHEST, MODE_ZEROING);
}

// Preserve the SDK's register type. The old name returns iter_reg while
// newer SDKs call the corresponding type addr_reg; neither is an integer.
// The unsigned argument is an element stride in a hardware loop, not a
// one-time pointer displacement. Leave loop-index scaling to the SDK.
__simd_callee__ inline auto update_addr_reg_b8(uint32_t offset) {
  return ::asc_create_iter_reg_b8(offset);
}
__simd_callee__ inline auto update_addr_reg_b16(uint32_t offset) {
  return ::asc_create_iter_reg_b16(offset);
}
__simd_callee__ inline auto update_addr_reg_b32(uint32_t offset) {
  return ::asc_create_iter_reg_b32(offset);
}

#define TL_CANN_LOAD(New, Old)                                                 \
  template <typename V, typename T, typename Offset>                           \
  __simd_callee__ inline void New(V &dst, __ubuf__ T *src, Offset offset) {    \
    ::Old(dst, src, offset);                                                   \
  }
#define TL_CANN_LOAD_UPDATE(New, Old)                                          \
  template <typename V, typename T>                                            \
  __simd_callee__ inline void New(V &dst, __ubuf__ T *&src, int32_t offset) {  \
    ::Old(dst, src, offset);                                                   \
  }
TL_CANN_LOAD(loadalign_brc_elem, asc_loadalign_brc)
TL_CANN_LOAD(loadalign_brc_datablock, asc_loadalign_brc_v2)
TL_CANN_LOAD(loadalign_brc_elem2datablock, asc_loadalign_brc_v3)
TL_CANN_LOAD(loadalign_unpack4, asc_loadalign_unpack_v2)
TL_CANN_LOAD_UPDATE(loadalign_brc_elem_postupdate, asc_loadalign_brc_postupdate)
TL_CANN_LOAD_UPDATE(loadalign_brc_datablock_postupdate,
                    asc_loadalign_brc_postupdate_v2)
TL_CANN_LOAD_UPDATE(loadalign_brc_elem2datablock_postupdate,
                    asc_loadalign_brc_postupdate_v3)
TL_CANN_LOAD_UPDATE(loadalign_unpack4_postupdate,
                    asc_loadalign_unpack_postupdate_v2)
#undef TL_CANN_LOAD
#undef TL_CANN_LOAD_UPDATE

template <typename D, typename S>
__simd_callee__ inline void pack_to_low(D &dst, S src) {
  ::asc_pack(dst, src);
}
template <typename D, typename S>
__simd_callee__ inline void pack_to_high(D &dst, S src) {
  ::asc_pack_v2(dst, src);
}
template <typename T, typename V, typename Offset>
__simd_callee__ inline void storealign_pack_quarter(__ubuf__ T *dst, V src,
                                                    Offset offset,
                                                    vector_bool mask) {
  ::asc_storealign_pack_v2(dst, src, offset, mask);
}

template <typename T, typename V>
__simd_callee__ inline void
storealign_postupdate(__ubuf__ T *&dst, V value, uint16_t block_stride,
                      uint16_t repeat_stride, vector_bool mask) {
  if ASC_IS_AIV {
    uint32_t config = (uint32_t(block_stride) << 16) | uint32_t(repeat_stride);
    if constexpr (sizeof(T) == 1) {
      auto physical = reinterpret_cast<__ubuf__ uint8_t *>(dst);
      ::vsstb(__builtin_bit_cast(vector_u8, value), physical,
              static_cast<int32_t>(config), mask, POST_UPDATE);
      dst = reinterpret_cast<__ubuf__ T *>(physical);
    } else {
      ::vsstb(value, dst, static_cast<int32_t>(config), mask, POST_UPDATE);
    }
  }
}
#else
template <typename V, typename S>
__simd_callee__ inline void min_scalar(V &dst, V src, S scalar,
                                       vector_bool mask) {
  ::asc_min_scalar(dst, src, scalar, mask);
}
template <typename V>
__simd_callee__ inline void ne(vector_bool &dst, V lhs, V rhs,
                               vector_bool mask) {
  ::asc_ne(dst, lhs, rhs, mask);
}
template <typename V>
__simd_callee__ inline void duplicate(V &dst, V src, vector_bool mask) {
  ::asc_duplicate(dst, src, mask);
}
template <typename V>
__simd_callee__ inline void duplicate_highest(V &dst, V src, vector_bool mask) {
  ::asc_duplicate_highest(dst, src, mask);
}

__simd_callee__ inline auto update_addr_reg_b8(uint32_t offset) {
  return ::asc_update_addr_reg_b8(offset);
}
__simd_callee__ inline auto update_addr_reg_b16(uint32_t offset) {
  return ::asc_update_addr_reg_b16(offset);
}
__simd_callee__ inline auto update_addr_reg_b32(uint32_t offset) {
  return ::asc_update_addr_reg_b32(offset);
}

#define TL_CANN_NATIVE_LOAD(Name)                                              \
  template <typename V, typename T, typename Offset>                           \
  __simd_callee__ inline void Name(V &dst, __ubuf__ T *src, Offset offset) {   \
    ::asc_##Name(dst, src, offset);                                            \
  }                                                                            \
  template <typename V, typename T>                                            \
  __simd_callee__ inline void Name##_postupdate(V &dst, __ubuf__ T *&src,      \
                                                int32_t offset) {              \
    ::asc_##Name##_postupdate(dst, src, offset);                               \
  }
TL_CANN_NATIVE_LOAD(loadalign_brc_elem)
TL_CANN_NATIVE_LOAD(loadalign_brc_datablock)
TL_CANN_NATIVE_LOAD(loadalign_brc_elem2datablock)
TL_CANN_NATIVE_LOAD(loadalign_unpack4)
#undef TL_CANN_NATIVE_LOAD
template <typename D, typename S>
__simd_callee__ inline void pack_to_low(D &dst, S src) {
  ::asc_pack_to_low(dst, src);
}
template <typename D, typename S>
__simd_callee__ inline void pack_to_high(D &dst, S src) {
  ::asc_pack_to_high(dst, src);
}
template <typename T, typename V, typename Offset>
__simd_callee__ inline void storealign_pack_quarter(__ubuf__ T *dst, V src,
                                                    Offset offset,
                                                    vector_bool mask) {
  ::asc_storealign_pack_quarter(dst, src, offset, mask);
}
template <typename T, typename V>
__simd_callee__ inline void
storealign_postupdate(__ubuf__ T *&dst, V value, uint16_t block_stride,
                      uint16_t repeat_stride, vector_bool mask) {
  ::asc_storealign_postupdate(dst, value, block_stride, repeat_stride, mask);
}
#endif

#endif // register-vector API

#undef TL_CANN_9_1
#endif // defined(__NPU_ARCH__)

} // namespace cann
} // namespace tl
