#pragma once

#include "../common.h"
#include <cute/arch/cluster_sm90.hpp>

namespace tl {

#ifndef TL_ALWAYS_FALSE_V_DEFINED
#define TL_ALWAYS_FALSE_V_DEFINED
template <class> inline constexpr bool always_false_v = false;
#endif

// Generic declaration: unsupported by default
template <DataType C_type, bool use_2cta = false>
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
template <DataType C_type, bool use_2cta = false>
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
TL_DEVICE void tcgen05mma_ts<DataType::kFloat16, false>(
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

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat16, true>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  // NOTE(wt): 2cta tcgen05mma requires 8 masks.
  // To keep API compatible, we still pass 4 masks as 1cta for now, but don't
  // use them in PTX. We shall refactor this in the future.
  if (cute::elect_one_sync()) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], [%1], %2, %3, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(desc_val), "r"(scalec));
  }
}

// BF16 maps to the same f16-kind instruction
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kBFloat16, false>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat16, false>(
      tmem_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kBFloat16, true>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat16, true>(tmem_a, desc_b, tmem_c, scalec,
                                          desc_val, mask0, mask1, mask2, mask3);
}

// TF32 instruction kind (2cta not supported currently)
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

// INT8 instruction kind (maps to kind::i8)
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kInt8, false>(
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

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kInt8, true>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::i8 [%0], [%1], %2, %3, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec));
  }
}

// UINT8 maps to the same i8-kind instruction
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kUInt8, false>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kInt8, false>(tmem_a, desc_b, tmem_c, scalec,
                                        desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kUInt8, true>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kInt8, true>(tmem_a, desc_b, tmem_c, scalec, desc_val,
                                       mask0, mask1, mask2, mask3);
}

// FP8 family instruction kind (maps to kind::f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat8_e4m3, false>(
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
TL_DEVICE void tcgen05mma_ts<DataType::kFloat8_e4m3, true>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [%1], %2, %3, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "r"(tmem_a), "l"(desc_b), "r"(desc_val), "r"(scalec));
  }
}

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat8_e5m2, false>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat8_e4m3, false>(
      tmem_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat8_e5m2, true>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat8_e4m3, true>(
      tmem_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

// FP6 family instruction kind (maps to kind::f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat6_e2m3fn, false>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat8_e4m3, false>(
      tmem_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat6_e2m3fn, true>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat8_e4m3, true>(
      tmem_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat6_e3m2fn, false>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat8_e4m3, false>(
      tmem_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat6_e3m2fn, true>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat8_e4m3, true>(
      tmem_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

// FP4 family instruction kind (maps to kind::f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat4_e2m1fn, false>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat8_e4m3, false>(
      tmem_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ts<DataType::kFloat4_e2m1fn, true>(
    uint32_t const &tmem_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ts<DataType::kFloat8_e4m3, true>(
      tmem_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

// F16/BF16 instruction kind (maps to kind::f16)
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat16, false>(
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

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat16, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  // idescE upper 32 bits carry the instruction descriptor; lower 32 ignored for
  // SS Load TMEM base from shared memory slot handled by caller

  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec));
  }
}

// BF16 maps to the same f16-kind instruction
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kBFloat16, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat16, false>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kBFloat16, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat16, true>(desc_a, desc_b, tmem_c, scalec,
                                          desc_val, mask0, mask1, mask2, mask3);
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

// INT8 instruction kind (maps to kind::i8)
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kInt8, false>(
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

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kInt8, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::2.kind::i8 [%0], %1, %2, %3, p; \n\t"
                 "}\n"
                 :
                 : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val),
                   "r"(scalec));
  }
}

// UINT8 maps to the same i8-kind instruction
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kUInt8, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kInt8, false>(desc_a, desc_b, tmem_c, scalec,
                                        desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kUInt8, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kInt8, true>(desc_a, desc_b, tmem_c, scalec, desc_val,
                                       mask0, mask1, mask2, mask3);
}

// FP8 family instruction kind (maps to kind::f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat8_e4m3, false>(
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
TL_DEVICE void tcgen05mma_ss<DataType::kFloat8_e4m3, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  if (cute::elect_one_sync()) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val), "r"(scalec));
  }
}

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat8_e5m2, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat8_e4m3, false>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat8_e5m2, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat8_e4m3, true>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

// FP6 family instruction kind (maps to kind::f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat6_e2m3fn, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat8_e4m3, false>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat6_e2m3fn, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat8_e4m3, true>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat6_e3m2fn, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat8_e4m3, false>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat6_e3m2fn, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat8_e4m3, true>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

// FP4 family instruction kind (maps to kind::f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat4_e2m1fn, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat8_e4m3, false>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ss<DataType::kFloat4_e2m1fn, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ss<DataType::kFloat8_e4m3, true>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
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

// UINT8 ws, maps to the same i8-kind instruction
template <>
TL_DEVICE void tcgen05mma_ws_ss<DataType::kUInt8>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ws_ss<DataType::kInt8>(desc_a, desc_b, tmem_c, scalec, desc_val,
                                    mask0, mask1, mask2, mask3);
}

// FP8 ws (maps to kind::f8f6f4)
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

// FP6 ws (maps to kind::f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ws_ss<DataType::kFloat6_e2m3fn>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ws_ss<DataType::kFloat8_e4m3>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

template <>
TL_DEVICE void tcgen05mma_ws_ss<DataType::kFloat6_e3m2fn>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ws_ss<DataType::kFloat8_e4m3>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

// FP4 ws (maps to kind::f8f6f4)
template <>
TL_DEVICE void tcgen05mma_ws_ss<DataType::kFloat4_e2m1fn>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, int const &mask0,
    int const &mask1, int const &mask2, int const &mask3) {
  tcgen05mma_ws_ss<DataType::kFloat8_e4m3>(
      desc_a, desc_b, tmem_c, scalec, desc_val, mask0, mask1, mask2, mask3);
}

// ============================================================================
// Block-scaled MMA variants:
// tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale Used for MXFP8
// block-scaled GEMM with scale factors in TMEM.
// ============================================================================

// Generic declaration: unsupported by default
template <DataType C_type, bool use_2cta = false>
TL_DEVICE void tcgen05mma_blockscaled_ss(uint64_t const & /*desc_a*/,
                                         uint64_t const & /*desc_b*/,
                                         uint32_t const & /*tmem_c*/,
                                         uint32_t const & /*scalec*/,
                                         uint32_t const & /*desc_val*/,
                                         uint32_t const & /*tmem_sfa*/,
                                         uint32_t const & /*tmem_sfb*/) {
  static_assert(
      always_false_v<std::integral_constant<int, static_cast<int>(C_type)>>,
      "tl::tcgen05mma_blockscaled_ss: unsupported accumulator type");
}

// FP8 E4M3 block-scaled
template <>
TL_DEVICE void tcgen05mma_blockscaled_ss<DataType::kFloat8_e4m3, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, uint32_t const &tmem_sfa,
    uint32_t const &tmem_sfb) {
  if (cute::elect_one_sync()) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, "
        "%3, [%5], [%6], p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val), "r"(scalec),
          "r"(tmem_sfa), "r"(tmem_sfb));
  }
}

template <>
TL_DEVICE void tcgen05mma_blockscaled_ss<DataType::kFloat8_e4m3, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, uint32_t const &tmem_sfa,
    uint32_t const &tmem_sfb) {
  if (cute::elect_one_sync()) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [%0], %1, %2, "
        "%3, [%5], [%6], p; \n\t"
        "}\n"
        :
        : "r"(tmem_c), "l"(desc_a), "l"(desc_b), "r"(desc_val), "r"(scalec),
          "r"(tmem_sfa), "r"(tmem_sfb));
  }
}

// FP8 E5M2 maps to the same instruction
template <>
TL_DEVICE void tcgen05mma_blockscaled_ss<DataType::kFloat8_e5m2, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, uint32_t const &tmem_sfa,
    uint32_t const &tmem_sfb) {
  tcgen05mma_blockscaled_ss<DataType::kFloat8_e4m3, false>(
      desc_a, desc_b, tmem_c, scalec, desc_val, tmem_sfa, tmem_sfb);
}

template <>
TL_DEVICE void tcgen05mma_blockscaled_ss<DataType::kFloat8_e5m2, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, uint32_t const &tmem_sfa,
    uint32_t const &tmem_sfb) {
  tcgen05mma_blockscaled_ss<DataType::kFloat8_e4m3, true>(
      desc_a, desc_b, tmem_c, scalec, desc_val, tmem_sfa, tmem_sfb);
}

// FP6 block-scaled
template <>
TL_DEVICE void tcgen05mma_blockscaled_ss<DataType::kFloat6_e2m3fn, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, uint32_t const &tmem_sfa,
    uint32_t const &tmem_sfb) {
  tcgen05mma_blockscaled_ss<DataType::kFloat8_e4m3, false>(
      desc_a, desc_b, tmem_c, scalec, desc_val, tmem_sfa, tmem_sfb);
}

template <>
TL_DEVICE void tcgen05mma_blockscaled_ss<DataType::kFloat6_e2m3fn, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, uint32_t const &tmem_sfa,
    uint32_t const &tmem_sfb) {
  tcgen05mma_blockscaled_ss<DataType::kFloat8_e4m3, true>(
      desc_a, desc_b, tmem_c, scalec, desc_val, tmem_sfa, tmem_sfb);
}

template <>
TL_DEVICE void tcgen05mma_blockscaled_ss<DataType::kFloat6_e3m2fn, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, uint32_t const &tmem_sfa,
    uint32_t const &tmem_sfb) {
  tcgen05mma_blockscaled_ss<DataType::kFloat8_e4m3, false>(
      desc_a, desc_b, tmem_c, scalec, desc_val, tmem_sfa, tmem_sfb);
}

template <>
TL_DEVICE void tcgen05mma_blockscaled_ss<DataType::kFloat6_e3m2fn, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, uint32_t const &tmem_sfa,
    uint32_t const &tmem_sfb) {
  tcgen05mma_blockscaled_ss<DataType::kFloat8_e4m3, true>(
      desc_a, desc_b, tmem_c, scalec, desc_val, tmem_sfa, tmem_sfb);
}

// FP4 block-scaled
template <>
TL_DEVICE void tcgen05mma_blockscaled_ss<DataType::kFloat4_e2m1fn, false>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, uint32_t const &tmem_sfa,
    uint32_t const &tmem_sfb) {
  tcgen05mma_blockscaled_ss<DataType::kFloat8_e4m3, false>(
      desc_a, desc_b, tmem_c, scalec, desc_val, tmem_sfa, tmem_sfb);
}

template <>
TL_DEVICE void tcgen05mma_blockscaled_ss<DataType::kFloat4_e2m1fn, true>(
    uint64_t const &desc_a, uint64_t const &desc_b, uint32_t const &tmem_c,
    uint32_t const &scalec, uint32_t const &desc_val, uint32_t const &tmem_sfa,
    uint32_t const &tmem_sfb) {
  tcgen05mma_blockscaled_ss<DataType::kFloat8_e4m3, true>(
      desc_a, desc_b, tmem_c, scalec, desc_val, tmem_sfa, tmem_sfb);
}

} // namespace tl
