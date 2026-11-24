#pragma once

#include "c_api/asc_simd.h"
#include <stdint.h>
#include <type_traits>
#include <version/asc_devkit_version.h>

// Stable names used by Ascend codegen. asc_simd.h selects the SDK APIs
// available for the compilation target.
namespace tl {
namespace cann {

#define TL_CANN_9_1                                                            \
  (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR == 1 && ASC_DEVKIT_PATCH == 0)

#if !defined(__NPU_ARCH__) || TL_CANN_9_1
enum class hf32_round_mode { NEAREST_AWAY, NEAREST_EVEN };
#else
using hf32_round_mode = ::asc_hf32_round_mode;
#endif

#if defined(__NPU_ARCH__)
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
  ::asc_set_hf32_round_mode(mode);
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

// Some CANN 9.1 targets use a packed GM-to-L1 control word and longer
// L0C store signatures. Keep the extra store fields at their neutral values.
#if TL_CANN_9_1
__aicore__ inline uint64_t pack_gm2l1_nz_para(uint16_t batches,
                                              uint16_t n_stride,
                                              uint16_t c0_stride,
                                              uint16_t matrix_stride) {
  return uint64_t(batches) | (uint64_t(n_stride) << 16) |
         (uint64_t(c0_stride) << 32) | (uint64_t(matrix_stride) << 48);
}

template <typename D, typename S>
__aicore__ inline void
copy_l0c2gm_9_1(__gm__ D *dst, __cc__ S *src, uint16_t n, uint16_t m,
                uint32_t dst_stride, uint16_t src_stride,
                store_l2_cache_mode cache, unit_flag_mode unit,
                quant_mode quant, relu_pre_mode relu, bool split, bool nz2nd,
                bool nz2dn, bool clip) {
  asc_copy_l0c2gm(
      dst, src, n, m, dst_stride, src_stride, cache, clip, unit, quant, relu,
      split, nz2nd, /*quant_post=*/0, /*relu_post=*/0, /*clip_relu_post=*/false,
      /*eltwise_op=*/0, /*eltwise_antq_en=*/false, /*c0_pad_en=*/false,
      /*broadcast_en=*/false, nz2dn);
}

template <typename D, typename S>
__aicore__ inline void
copy_l0c2ub_9_1(__ubuf__ D *dst, __cc__ S *src, uint16_t n, uint16_t m,
                uint32_t dst_stride, uint16_t src_stride, int8_t subblock,
                dual_dst_mode dual, unit_flag_mode unit, quant_mode quant,
                relu_pre_mode relu, bool split, bool nz2nd, bool nz2dn,
                bool clip) {
  asc_copy_l0c2ub(
      dst, src, n, m, dst_stride, src_stride, dual, bool(subblock), clip, unit,
      quant, relu, split, nz2nd, /*quant_post=*/0, /*relu_post=*/0,
      /*clip_relu_post=*/false, /*eltwise_op=*/0, /*eltwise_antq_en=*/false,
      /*c0_pad_en=*/false, /*broadcast_en=*/false, nz2dn);
}
#endif

template <typename... Args>
__aicore__ inline void set_gm2l1_nz_para(Args... args) {
  if constexpr (TL_CANN_9_1 && !requires { asc_set_gm2l1_nz_para(args...); }) {
    asc_set_gm2l1_nz_para(pack_gm2l1_nz_para(args...));
  } else {
    asc_set_gm2l1_nz_para(args...);
  }
}

template <typename... Args>
__aicore__ inline void set_l0c_copy_nz_para(Args... args) {
  if constexpr (TL_CANN_9_1 &&
                !requires { asc_set_l0c_copy_nz_para(args...); }) {
    asc_set_l0c2gm_nz2nd(args...);
  } else {
    asc_set_l0c_copy_nz_para(args...);
  }
}

template <typename... Args> __aicore__ inline void copy_l0c2gm(Args... args) {
  if constexpr (TL_CANN_9_1 && !requires { asc_copy_l0c2gm(args...); }) {
    copy_l0c2gm_9_1(args...);
  } else {
    asc_copy_l0c2gm(args...);
  }
}

template <typename... Args> __aicore__ inline void copy_l0c2ub(Args... args) {
  if constexpr (TL_CANN_9_1 && !requires { asc_copy_l0c2ub(args...); }) {
    copy_l0c2ub_9_1(args...);
  } else {
    asc_copy_l0c2ub(args...);
  }
}

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
__simd_callee__ inline void select(V &dst, V lhs, V rhs, vector_bool mask) {
  if constexpr (requires { asc_select(dst, lhs, rhs, mask); }) {
    asc_select(dst, lhs, rhs, mask);
  } else {
    // Bisheng has a typed vsel even for E8M0, which lacks an SDK overload.
    vsel(dst, lhs, rhs, mask);
  }
}

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

template <typename T, typename V>
__simd_callee__ inline void
storealign_postupdate(__ubuf__ T *&dst, V value, uint16_t block_stride,
                      uint16_t repeat_stride, vector_bool mask) {
  ::asc_storealign_postupdate(dst, value, block_stride, repeat_stride, mask);
}
#endif

// These APIs were renamed. Forward the whole overload set, including loads
// that return a vector and post-update operations that write back a pointer.
#if TL_CANN_9_1
#define TL_CANN_FORWARD_REG(Name, Old) TL_CANN_FORWARD_TO(Name, Old)
#else
#define TL_CANN_FORWARD_REG(Name, Old) TL_CANN_FORWARD_TO(Name, asc_##Name)
#endif
#define TL_CANN_FORWARD_TO(Name, Native)                                       \
  template <typename... Args>                                                  \
  __simd_callee__ inline decltype(auto) Name(Args &&...args) {                 \
    return ::Native(static_cast<Args &&>(args)...);                            \
  }
TL_CANN_FORWARD_REG(update_addr_reg_b8, asc_create_iter_reg_b8)
TL_CANN_FORWARD_REG(update_addr_reg_b16, asc_create_iter_reg_b16)
TL_CANN_FORWARD_REG(update_addr_reg_b32, asc_create_iter_reg_b32)
TL_CANN_FORWARD_REG(loadalign_brc_elem, asc_loadalign_brc)
TL_CANN_FORWARD_REG(loadalign_brc_datablock, asc_loadalign_brc_v2)
TL_CANN_FORWARD_REG(loadalign_brc_elem2datablock, asc_loadalign_brc_v3)
TL_CANN_FORWARD_REG(loadalign_unpack4, asc_loadalign_unpack_v2)
TL_CANN_FORWARD_REG(loadalign_brc_elem_postupdate, asc_loadalign_brc_postupdate)
TL_CANN_FORWARD_REG(loadalign_brc_datablock_postupdate,
                    asc_loadalign_brc_postupdate_v2)
TL_CANN_FORWARD_REG(loadalign_brc_elem2datablock_postupdate,
                    asc_loadalign_brc_postupdate_v3)
TL_CANN_FORWARD_REG(loadalign_unpack4_postupdate,
                    asc_loadalign_unpack_postupdate_v2)
TL_CANN_FORWARD_REG(pack_to_low, asc_pack)
TL_CANN_FORWARD_REG(pack_to_high, asc_pack_v2)
TL_CANN_FORWARD_REG(storealign_pack_quarter, asc_storealign_pack_v2)
#undef TL_CANN_FORWARD_TO
#undef TL_CANN_FORWARD_REG

#if TL_CANN_9_1
// 9.1 only provides the output-parameter forms of these loads and packs.
// Deduce the destination register from the matching SDK signature. In
// particular, packing an int16 register produces a uint8 register.
namespace detail {
template <typename Src> struct RegisterResult {
  template <typename Dst>
  static __simd_callee__ Dst type(__simd_callee__ void (*)(Dst &, Src));
};
} // namespace detail
#define TL_CANN_RETURN_REG(Name, Native)                                       \
  template <typename Src> __simd_callee__ inline auto Name(Src src) {          \
    decltype(detail::RegisterResult<Src>::type(::Native)) dst;                 \
    ::Native(dst, src);                                                        \
    return dst;                                                                \
  }
TL_CANN_RETURN_REG(loadalign_brc_elem, asc_loadalign_brc)
TL_CANN_RETURN_REG(loadalign_brc_datablock, asc_loadalign_brc_v2)
TL_CANN_RETURN_REG(loadalign_brc_elem2datablock, asc_loadalign_brc_v3)
TL_CANN_RETURN_REG(loadalign_unpack4, asc_loadalign_unpack_v2)
TL_CANN_RETURN_REG(pack_to_low, asc_pack)
TL_CANN_RETURN_REG(pack_to_high, asc_pack_v2)
#undef TL_CANN_RETURN_REG
#endif

// Register APIs also accept value-returning calls. Use the same corrected
// operation so the two SDK spellings agree on inactive lanes and edge types.
template <typename V>
__simd_callee__ inline V div(V lhs, V rhs, vector_bool mask) {
  V dst;
  div(dst, lhs, rhs, mask);
  return dst;
}
template <typename V>
__simd_callee__ inline V select(V lhs, V rhs, vector_bool mask) {
  V dst;
  select(dst, lhs, rhs, mask);
  return dst;
}
template <typename V, typename S>
__simd_callee__ inline V min_scalar(V src, S scalar, vector_bool mask) {
  V dst;
  min_scalar(dst, src, scalar, mask);
  return dst;
}
template <typename V>
__simd_callee__ inline vector_bool ne(V lhs, V rhs, vector_bool mask) {
  vector_bool dst;
  ne(dst, lhs, rhs, mask);
  return dst;
}
template <typename V>
__simd_callee__ inline V duplicate(V src, vector_bool mask) {
  V dst;
  duplicate(dst, src, mask);
  return dst;
}

// Only the five-argument strided store needs the 9.1 instruction fallback.
template <typename... Args>
__simd_callee__ inline void storealign_postupdate(Args &&...args)
    requires(sizeof...(Args) != 5) {
  ::asc_storealign_postupdate(static_cast<Args &&>(args)...);
}

#endif // register-vector API

#endif // defined(__NPU_ARCH__)
#undef TL_CANN_9_1

} // namespace cann
} // namespace tl
