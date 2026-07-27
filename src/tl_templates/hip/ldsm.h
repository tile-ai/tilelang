#pragma once

#include "common.h"

namespace tl {

#if defined(__gfx950__)

// ds_read_tr16_b64: LDS transpose read, 64-bit, 16-element transpose.
// Reads 8 bytes from LDS with a transpose across 16 elements.
// Used for FP16/BF16 MFMA matrix B loads on gfx950 (MI350/MI355X).
// smem_ptr must point into __shared__ memory.
//
// Uses __builtin_amdgcn_ds_read_tr16_b64_v4f16 (LLVM builtin) instead of
// inline assembly because ROCm <= 7.2 assembler does not yet recognise the
// ds_read_tr16_b64 mnemonic even though the hardware supports it.
CK_TILE_DEVICE uint2 ds_read_tr16_b64(const void *smem_ptr) {
  typedef __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 fp16x4_t;
  // C-style cast: void* → LDS fp16x4_t* (required by the LLVM builtin
  // signature)
  fp16x4_t v = __builtin_amdgcn_ds_read_tr16_b64_v4f16(
      (__attribute__((address_space(3))) fp16x4_t *)(smem_ptr));
  uint2 result;
  __builtin_memcpy(&result, &v, sizeof(result));
  return result;
}

// ds_read_tr8_b64: LDS transpose read, 64-bit, 8-element transpose.
// Reads 8 bytes from LDS with a transpose across 8 elements.
// Used for FP32 MFMA matrix B loads on gfx950 (MI350/MI355X).
// smem_ptr must point into __shared__ memory.
//
// Uses __builtin_amdgcn_ds_read_tr8_b64_v2i32 (LLVM builtin) for the same
// reason as ds_read_tr16_b64 above.
CK_TILE_DEVICE uint2 ds_read_tr8_b64(const void *smem_ptr) {
  typedef __attribute__((__vector_size__(2 * sizeof(int)))) int i32x2_t;
  i32x2_t v = __builtin_amdgcn_ds_read_tr8_b64_v2i32(
      (__attribute__((address_space(3))) i32x2_t *)(smem_ptr));
  uint2 result;
  __builtin_memcpy(&result, &v, sizeof(result));
  return result;
}

#endif // __gfx950__

} // namespace tl
