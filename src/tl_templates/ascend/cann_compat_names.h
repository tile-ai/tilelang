#pragma once

// Native SDK names shared by device lookup, template macros and Ascend codegen.

#define TL_CANN_CORE_NAMES(X)                                                  \
  X(disable_hf32)                                                              \
  X(set_hf32_round_mode)                                                       \
  X(set_atomic_add_int32)                                                      \
  X(set_atomic_max_int32)                                                      \
  X(set_atomic_min_int32)                                                      \
  X(load_dev)                                                                  \
  X(set_gm2l1_nz_para)                                                         \
  X(set_l0c_copy_nz_para)                                                      \
  X(copy_l0c2gm)                                                               \
  X(copy_l0c2ub)                                                               \
  X(copy_gm2ub_align)                                                          \
  X(copy_ub2gm_align)

#define TL_CANN_SIMD_NAMES(X)                                                  \
  X(div)                                                                       \
  X(min_scalar)                                                                \
  X(ne)                                                                        \
  X(select)                                                                    \
  X(duplicate)                                                                 \
  X(duplicate_highest)                                                         \
  X(update_addr_reg_b8)                                                        \
  X(update_addr_reg_b16)                                                       \
  X(update_addr_reg_b32)                                                       \
  X(loadalign_brc_elem)                                                        \
  X(loadalign_brc_elem_postupdate)                                             \
  X(loadalign_brc_datablock)                                                   \
  X(loadalign_brc_datablock_postupdate)                                        \
  X(loadalign_brc_elem2datablock)                                              \
  X(loadalign_brc_elem2datablock_postupdate)                                   \
  X(loadalign_unpack4)                                                         \
  X(loadalign_unpack4_postupdate)                                              \
  X(pack_to_low)                                                               \
  X(pack_to_high)                                                              \
  X(storealign_pack_quarter)                                                   \
  X(storealign_postupdate)

#define TL_CANN_TYPE_NAMES(X)                                                  \
  X(hf32_round_mode)                                                           \
  X(load_l2_cache_mode)                                                        \
  X(store_l2_cache_mode)                                                       \
  X(unit_flag_mode)                                                            \
  X(quant_mode)                                                                \
  X(relu_pre_mode)                                                             \
  X(dual_dst_mode)

#define TL_CANN_SIMT_MATH_NAMES(X)                                             \
  X(acosf)                                                                     \
  X(acoshf)                                                                    \
  X(asinf)                                                                     \
  X(asinhf)                                                                    \
  X(atanf)                                                                     \
  X(atan2f)                                                                    \
  X(atanhf)                                                                    \
  X(copysignf)                                                                 \
  X(coshf)                                                                     \
  X(exp2f)                                                                     \
  X(exp10f)                                                                    \
  X(fmodf)                                                                     \
  X(hypotf)                                                                    \
  X(log1pf)                                                                    \
  X(powf)                                                                      \
  X(sinhf)                                                                     \
  X(tanhf)                                                                     \
  X(hexp2)                                                                     \
  X(hexp10)
