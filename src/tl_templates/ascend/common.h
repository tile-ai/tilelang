#pragma once

#include <cstdint>
#include <stdio.h>

#include "acl/acl.h"
#include "c_api/asc_simd.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_fp8.h"
#include "simt_api/asc_simt.h"
#include "tl_templates/ascend/ascend_fp8.h"
#include "tl_templates/ascend/numeric_limits.h"
#include "tl_templates/ascend/reduce.h"
#include "tl_templates/ascend/simd_inst.h"

// Use standard unsigned types instead of macros to avoid conflicts with system
// headers
typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

namespace tl {

__aicore__ inline uint32_t scalar_mulhi(uint32_t a, uint32_t b) {
  return static_cast<uint32_t>((uint64_t(a) * uint64_t(b)) >> 32);
}

__aicore__ inline uint64_t scalar_mulhi(uint64_t a, uint64_t b) {
  // Base-2^32 multiplication: carry the middle three limbs into the high word.
  const uint64_t mask = 0xffffffffULL;
  const uint64_t a0 = a & mask, a1 = a >> 32;
  const uint64_t b0 = b & mask, b1 = b >> 32;
  const uint64_t p00 = a0 * b0, p01 = a0 * b1;
  const uint64_t p10 = a1 * b0, p11 = a1 * b1;
  const uint64_t carry = ((p00 >> 32) + (p01 & mask) + (p10 & mask)) >> 32;
  return p11 + (p01 >> 32) + (p10 >> 32) + carry;
}

} // namespace tl

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    aclError __err = (stmt);                                                   \
    if (__err != ACL_SUCCESS) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: aclError %d", __FILE__,      \
               __LINE__, (int)__err);                                          \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    auto __err = aclrtGetLastError(ACL_RT_THREAD_LEVEL);                       \
    if (__err) {                                                               \
      snprintf(error_buf, ERROR_BUF_SIZE, kernel_name ": aclError %d",         \
               (int)__err);                                                    \
      return -1;                                                               \
    }                                                                          \
  } while (0)
