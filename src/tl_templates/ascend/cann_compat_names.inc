// The emitter consumes this list as a lookup table. Device headers consume it
// with TL_CANN_SCOPE set to push_macro or pop_macro to delimit native names.
#ifdef TL_CANN_SCOPE
#define TL_CANN_PRAGMA_IMPL(x) _Pragma(#x)
#define TL_CANN_PRAGMA(x) TL_CANN_PRAGMA_IMPL(x)
#define TL_CANN_SCOPE_NAME(name) TL_CANN_PRAGMA(TL_CANN_SCOPE(#name))
#define TL_CANN_CORE(name) TL_CANN_SCOPE_NAME(asc_##name)
#define TL_CANN_TYPE(name) TL_CANN_CORE(name)
#ifdef TL_CANN_SIMD_SCOPE
#define TL_CANN_SIMD(name) TL_CANN_CORE(name)
#else
#define TL_CANN_SIMD(name)
#endif
#endif

// clang-format off
TL_CANN_CORE(disable_hf32)
TL_CANN_CORE(set_hf32_round_mode)
TL_CANN_CORE(set_atomic_add_int32)
TL_CANN_CORE(set_atomic_max_int32)
TL_CANN_CORE(set_atomic_min_int32)
TL_CANN_CORE(load_dev)
TL_CANN_CORE(set_gm2l1_nz_para)
TL_CANN_CORE(set_l0c_copy_nz_para)
TL_CANN_CORE(copy_l0c2gm)
TL_CANN_CORE(copy_l0c2ub)
TL_CANN_CORE(copy_gm2ub_align)
TL_CANN_CORE(copy_ub2gm_align)

TL_CANN_SIMD(div)
TL_CANN_SIMD(min_scalar)
TL_CANN_SIMD(ne)
TL_CANN_SIMD(select)
TL_CANN_SIMD(duplicate)
TL_CANN_SIMD(duplicate_highest)
TL_CANN_SIMD(update_addr_reg_b8)
TL_CANN_SIMD(update_addr_reg_b16)
TL_CANN_SIMD(update_addr_reg_b32)
TL_CANN_SIMD(loadalign_brc_elem)
TL_CANN_SIMD(loadalign_brc_elem_postupdate)
TL_CANN_SIMD(loadalign_brc_datablock)
TL_CANN_SIMD(loadalign_brc_datablock_postupdate)
TL_CANN_SIMD(loadalign_brc_elem2datablock)
TL_CANN_SIMD(loadalign_brc_elem2datablock_postupdate)
TL_CANN_SIMD(loadalign_unpack4)
TL_CANN_SIMD(loadalign_unpack4_postupdate)
TL_CANN_SIMD(pack_to_low)
TL_CANN_SIMD(pack_to_high)
TL_CANN_SIMD(storealign_pack_quarter)
TL_CANN_SIMD(storealign_postupdate)

TL_CANN_TYPE(hf32_round_mode)
TL_CANN_TYPE(load_l2_cache_mode)
TL_CANN_TYPE(store_l2_cache_mode)
TL_CANN_TYPE(unit_flag_mode)
TL_CANN_TYPE(quant_mode)
TL_CANN_TYPE(relu_pre_mode)
TL_CANN_TYPE(dual_dst_mode)
// clang-format on

#ifdef TL_CANN_SCOPE
#undef TL_CANN_CORE
#undef TL_CANN_SIMD
#undef TL_CANN_TYPE
#undef TL_CANN_SCOPE_NAME
#undef TL_CANN_PRAGMA
#undef TL_CANN_PRAGMA_IMPL
#endif
