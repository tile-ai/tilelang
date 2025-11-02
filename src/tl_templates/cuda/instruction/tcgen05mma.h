#pragma once

#include "../common.h"
#include <cute/arch/cluster_sm90.hpp>

namespace tl {

#ifndef TL_ALWAYS_FALSE_V_DEFINED
#define TL_ALWAYS_FALSE_V_DEFINED
template <class> inline constexpr bool always_false_v = false;
#endif

// Generic declaration: unsupported by default
template <DataType C_type>
TL_DEVICE void
tcgen05mma_ss(uint64_t const & /*desc_a*/, uint64_t const & /*desc_b*/,
              uint32_t const & /*tmem_c*/, uint32_t const & /*scalec*/,
              uint32_t const & /*desc_val*/, int const & /*mask0*/,
              int const & /*mask1*/, int const & /*mask2*/,
              int const & /*mask3*/) {
  static_assert(
      always_false_v<std::integral_constant<int, static_cast<int>(C_type)>>,
      "tl::tcgen05mma_ss: unsupported accumulator type");
}

// TS variants: A from TMEM, B from SMEM (desc)
// Generic declaration: unsupported by default
template <DataType C_type>
TL_DEVICE void
tcgen05mma_ts(uint32_t const & /*tmem_a*/, uint64_t const & /*desc_b*/,
              uint32_t const & /*tmem_c*/, uint32_t const & /*scalec*/,
              uint32_t const & /*desc_val*/, int const & /*mask0*/,
              int const & /*mask1*/, int const & /*mask2*/,
              int const & /*mask3*/) {
  static_assert(
      always_false_v<std::integral_constant<int, static_cast<int>(C_type)>>,
      "tl::tcgen05mma_ts: unsupported accumulator type");
}

// F16/BF16 instruction kind (maps to kind::f16)
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat16>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, {%5, "
                 "%6, %7, %8}, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3));
  }
}

// BF16 maps to the same f16-kind instruction
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kBFloat16>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat16>(tmem_a, desc_b, tmem_c, scalec, desc_val,
                                    mask0, mask1, mask2, mask3);
}

// TF32 instruction kind
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kTensorFloat32>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], %2, %3, {%5, "
                 "%6, %7, %8}, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3));
  }
}

// INT8 instruction kind
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kInt8>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::i8 [%0], [%1], %2, %3, {%5, "
                 "%6, %7, %8}, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3));
  }
}

// FP8 family instruction kind (maps to f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat8_e4m3>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], %2, %3, "
                 "{%5, %6, %7, %8}, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3));
  }
}

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat8_e5m2>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat8_e4m3>(tmem_a, desc_b, tmem_c, scalec,
                                        desc_val, mask0, mask1, mask2, mask3);
}

// F16/BF16 instruction kind (maps to kind::f16)
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat16>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  // idescE upper 32 bits carry the instruction descriptor; lower 32 ignored for
  // SS Load TMEM base from shared memory slot handled by caller

  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, "
                 "%6, %7, %8}, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3));
  }
}

// BF16 maps to the same f16-kind instruction
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kBFloat16>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat16>(desc_a, desc_b, tmem_c, scalec, desc_val,
                                    mask0, mask1, mask2, mask3);
}

// TF32 instruction kind
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kTensorFloat32>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, {%5, "
                 "%6, %7, %8}, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3));
  }
}

// INT8 instruction kind
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kInt8>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, {%5, %6, "
                 "%7, %8}, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3));
  }
}

// FP8 family instruction kind (maps to f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat8_e4m3>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, "
                 "%6, %7, %8}, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec), "r"(mask0), "r"(mask1), "r"(mask2), "r"(mask3));
  }
}

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat8_e5m2>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat8_e4m3>(desc_a, desc_b, tmem_c, scalec,
                                        desc_val, mask0, mask1, mask2, mask3);
}

// WS variants: tcgen05.mma.ws.cta_group::1.kind::xxx
// Generic declaration falls back to static assert
template <DataType C_type>
TL_DEVICE void
tcgen05mma_ws_ss(uint64_t const & /*desc_a*/, uint64_t const & /*desc_b*/,
                 uint32_t const & /*tmem_c*/, uint32_t const & /*scalec*/,
                 uint32_t const & /*desc_val*/, int const & /*mask0*/,
                 int const & /*mask1*/, int const & /*mask2*/,
                 int const & /*mask3*/) {
  static_assert(
      always_false_v<std::integral_constant<int, static_cast<int>(C_type)>>,
      "tl::tcgen05mma_ws_ss: unsupported accumulator type");
}

// F16/BF16 ws
template <>
TL_DEVICE void tcgen05mma_ws_ss<DataType::kFloat16>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, p, 0; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val), "r"(scalec));
  }
}

template <>
TL_DEVICE void tcgen05mma_ws_ss<DataType::kBFloat16>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ws_ss<DataType::kFloat16>(desc_a, desc_b, tmem_c, scalec, desc_val,
                                       mask0, mask1, mask2, mask3);
}

// TF32 ws
template <>
TL_DEVICE void tcgen05mma_ws_ss<DataType::kTensorFloat32>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::tf32 [%0], %1, %2, %3, p, 0; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val), "r"(scalec));
  }
}

// INT8 ws
template <>
TL_DEVICE void tcgen05mma_ws_ss<DataType::kInt8>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::i8 [%0], %1, %2, %3, p, 0; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val), "r"(scalec));
  }
}

// FP8 ws (maps to f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ws_ss<DataType::kFloat8_e4m3>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.ws.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p, 0; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val), "r"(scalec));
  }
}

template <>
TL_DEVICE void tcgen05mma_ws_ss<DataType::kFloat8_e5m2>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ws_ss<DataType::kFloat8_e4m3>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

} // namespace tl
