#pragma once

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
