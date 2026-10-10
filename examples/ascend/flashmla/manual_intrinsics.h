// Low-level instruction adapters for the manually scheduled FlashMLA example.
// Algorithm and memory plan: deepseek-ai/FlashMLA PR #229 (MIT license).
#pragma once
#include "c_api/asc_simd.h"
#include "kernel_operator.h"

__aicore__ inline void fm_init() {
  AscendC::InitSocState();
  asc_set_gm2ub_loop_size(1ul, 1ul);
  asc_set_ub2gm_loop_size(1ul, 1ul);
  asc_set_ctrl(0ull);
  asc_set_gm2l1_nz_para(uint64_t(1) | (uint64_t(1) << 16) |
                        (uint64_t(32) << 32));
  asc_set_l0c2gm_nz2nd(1u, 0u, 0u);
  asc_set_l3d_rpt_b(1 << 16);
}
__aicore__ inline void fm_ss_set(uint32_t slot, uint32_t sid, uint32_t value) {
  auto p = (volatile __ssbuf__ uint32_t *)AscendC::GetSsbufBaseAddr();
  p[slot * 2 + sid] = value;
}
__aicore__ inline uint32_t fm_ss_get(uint32_t slot) {
  auto p = (volatile __ssbuf__ uint64_t *)AscendC::GetSsbufBaseAddr();
  return p[slot] != 0;
}
__aicore__ inline void fm_q_load(__cbuf__ bfloat16_t *dst,
                                 __gm__ bfloat16_t *src, uint32_t stride) {
  asc_copy_gm2l1_nd2nz(dst, src, stride, 0u, 32u, 512u, 0u, false);
}
__aicore__ inline void fm_pair(__ubuf__ bfloat16_t *dst, __gm__ bfloat16_t *src,
                               uint32_t count, int64_t stride) {
  asc_copy_gm2ub_align(dst, src, count, 1024u, 0, 0, false,
                       asc_load_l2_cache_mode::NORMAL_LAST_VICTIM, stride,
                       1024ul);
}
__aicore__ inline void fm_kv_push(__cbuf__ bfloat16_t *dst,
                                  __ubuf__ bfloat16_t *src) {
  asc_copy_ub2l1(dst, src, 32u, 32u, 1u, 0u);
}
__aicore__ inline void fm_s_push(__cbuf__ bfloat16_t *dst,
                                 __ubuf__ bfloat16_t *src) {
  asc_copy_ub2l1(dst, src, 4u, 32u, 32u, 32u);
}
__aicore__ inline void fm_a_load(__ca__ bfloat16_t *dst,
                                 __cbuf__ bfloat16_t *src, uint32_t k) {
  asc_copy_l12l0a(dst, src, 0, k * 8u, 2u, 8u, 2u, 4u);
}
__aicore__ inline void fm_k_load(__cb__ bfloat16_t *dst,
                                 __cbuf__ bfloat16_t *src, uint32_t k) {
  asc_copy_l12l0b(dst, src, 0, k * 8u, 2u, 8u, 2u, 4u);
}
__aicore__ inline void fm_v_load(__cb__ bfloat16_t *dst,
                                 __cbuf__ bfloat16_t *src, uint32_t k) {
  asc_copy_l12l0b_transpose(dst, src, 0, k * 8u, 2u, 8u, 2u, 8u);
}
__aicore__ inline void fm_s_load(__ca__ bfloat16_t *dst,
                                 __cbuf__ bfloat16_t *src) {
  asc_copy_l12l0a(dst, src, 0, 0, 4u, 4u, 4u, 4u);
}
__aicore__ inline void fm_mmad(__cc__ float *dst, __ca__ bfloat16_t *a,
                               __cb__ bfloat16_t *b, uint32_t n, uint32_t k,
                               uint32_t unit, bool clear) {
  asc_mmad(dst, a, b, 64u, k, n, unit, true, false, clear);
}
__aicore__ inline void fm_p_store(__ubuf__ float *dst, __cc__ float *src) {
  asc_copy_l0c2ub(dst, src, 64u, 64u, 512u, 64u, 1, false, 0, 3, 0, 0, false,
                  false, 0, 0, 0, 0, 0, 0, 0, 0);
}
__aicore__ inline void fm_o_store(__ubuf__ float *dst, __cc__ float *src,
                                  uint32_t unit) {
  asc_copy_l0c2ub(dst, src, 128u, 64u, 128u, 64u, 1, false, 0, unit, 0, 0,
                  false, true, 0, 0, 0, 0, 0, 0, 0, 0);
}
__aicore__ inline void fm_write_o(__gm__ bfloat16_t *dst,
                                  __ubuf__ bfloat16_t *src) {
  asc_copy_ub2gm_align(dst, src, 32u, 256u,
                       asc_store_l2_cache_mode::NORMAL_FIRST_VICTIM, 1024ul,
                       2048ul);
}
__aicore__ inline void fm_pair_bytes(__ubuf__ uint8_t *dst, __gm__ uint8_t *src,
                                     uint32_t count, int64_t stride,
                                     uint32_t bytes, uint32_t pitch,
                                     uint32_t cache) {
  asc_copy_gm2ub_align(dst, src, count, bytes, 0, 0, false,
                       cache ? asc_load_l2_cache_mode::NORMAL_LAST_VICTIM
                             : asc_load_l2_cache_mode::NOTALLOC_CLEAN,
                       stride, pitch);
}
