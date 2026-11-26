#pragma once

// Device templates include the SDK version header before using these checks.
#define TL_CANN_VERSION_SERIES(major, minor)                                   \
  (ASC_DEVKIT_MAJOR == (major) && ASC_DEVKIT_MINOR == (minor))
#define TL_CANN_VERSION_EXACT(major, minor, patch)                             \
  (TL_CANN_VERSION_SERIES(major, minor) && ASC_DEVKIT_PATCH == (patch))

#define TL_CANN_9_1 TL_CANN_VERSION_SERIES(9, 1)
#define TL_CANN_9_2_0_OR_9_3_0                                                 \
  (TL_CANN_VERSION_EXACT(9, 2, 0) || TL_CANN_VERSION_EXACT(9, 3, 0))

// Native SDK names shared by device templates and Ascend codegen.

// Fixed-arity core APIs only route explicit global calls with that arity.
#define TL_CANN_CORE_NAMES(X)                                                  \
  X(disable_hf32, 0)                                                           \
  X(set_hf32_round_mode, 1)                                                    \
  X(set_atomic_add_int32, 0)                                                   \
  X(set_atomic_max_int32, 0)                                                   \
  X(set_atomic_min_int32, 0)                                                   \
  X(load_dev, 1)                                                               \
  X(set_gm2l1_nz_para)                                                         \
  X(set_l0c_copy_nz_para)                                                      \
  X(copy_l0c2gm)                                                               \
  X(copy_l0c2ub)                                                               \
  X(copy_gm2ub_align)                                                          \
  X(copy_ub2gm_align)

// Each renamed register API has a public name and a CANN 9.1 name.
// WITH_RESULT also supplies the missing value-returning overload.
// Generated VF code can call asc_update_addr_reg_b16 directly, while simd_inst
// helpers call names such as asc_loadalign_brc_elem. Both use these mappings.
// Pass names directly so consumers can paste asc_ before import macros expand.
#define TL_CANN_RENAMED_REG_NAMES(FORWARD, WITH_RESULT)                        \
  FORWARD(update_addr_reg_b8, asc_create_iter_reg_b8)                          \
  FORWARD(update_addr_reg_b16, asc_create_iter_reg_b16)                        \
  FORWARD(update_addr_reg_b32, asc_create_iter_reg_b32)                        \
  WITH_RESULT(loadalign_brc_elem, asc_loadalign_brc)                           \
  FORWARD(loadalign_brc_elem_postupdate, asc_loadalign_brc_postupdate)         \
  WITH_RESULT(loadalign_brc_datablock, asc_loadalign_brc_v2)                   \
  FORWARD(loadalign_brc_datablock_postupdate, asc_loadalign_brc_postupdate_v2) \
  WITH_RESULT(loadalign_brc_elem2datablock, asc_loadalign_brc_v3)              \
  FORWARD(loadalign_brc_elem2datablock_postupdate,                             \
          asc_loadalign_brc_postupdate_v3)                                     \
  WITH_RESULT(loadalign_unpack4, asc_loadalign_unpack_v2)                      \
  FORWARD(loadalign_unpack4_postupdate, asc_loadalign_unpack_postupdate_v2)    \
  WITH_RESULT(pack_to_low, asc_pack)                                           \
  WITH_RESULT(pack_to_high, asc_pack_v2)                                       \
  FORWARD(storealign_pack_quarter, asc_storealign_pack_v2)

#define TL_CANN_SIMD_NAMES(X)                                                  \
  X(div)                                                                       \
  X(min_scalar)                                                                \
  X(ne)                                                                        \
  X(select)                                                                    \
  X(duplicate)                                                                 \
  X(duplicate_highest)                                                         \
  TL_CANN_RENAMED_REG_NAMES(X, X)                                              \
  X(storealign_postupdate)

#define TL_CANN_TYPE_NAMES(X)                                                  \
  X(hf32_round_mode)                                                           \
  X(load_l2_cache_mode)                                                        \
  X(store_l2_cache_mode)                                                       \
  X(unit_flag_mode)                                                            \
  X(quant_mode)                                                                \
  X(relu_pre_mode)                                                             \
  X(dual_dst_mode)

// These functions use the native overloads outside CANN 9.1.
#define TL_CANN_SIMT_MATH_9_1_ONLY_NAMES(X)                                    \
  X(acosf)                                                                     \
  X(acoshf)                                                                    \
  X(asinf)                                                                     \
  X(asinhf)                                                                    \
  X(atanf)                                                                     \
  X(atanhf)                                                                    \
  X(copysignf)                                                                 \
  X(coshf)                                                                     \
  X(exp2f)                                                                     \
  X(exp10f)                                                                    \
  X(fmodf)                                                                     \
  X(hypotf)                                                                    \
  X(log1pf)                                                                    \
  X(sinhf)                                                                     \
  X(tanhf)                                                                     \
  X(hexp2)                                                                     \
  X(hexp10)

#define TL_CANN_SIMT_MATH_NAMES(X)                                             \
  TL_CANN_SIMT_MATH_9_1_ONLY_NAMES(X)                                          \
  X(atan2f)                                                                    \
  X(powf)

// SDK spellings with an asc_ prefix that a caller's macro could replace.
#define TL_CANN_SDK_NAMES(X)                                                   \
  TL_CANN_CORE_NAMES(X) TL_CANN_TYPE_NAMES(X) TL_CANN_SIMD_NAMES(X)
