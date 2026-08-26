#pragma once

#include "common.h"

namespace tl {

TL_DEVICE void ptx_ldmatrix_x1(void const *const smem_ptr,
                               void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
               : "=r"(value[0])
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_x2(void const *const smem_ptr,
                               void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
               : "=r"(value[0]), "=r"(value[1])
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_x4(void const *const smem_ptr,
                               void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
      : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_x1_trans(void const *const smem_ptr,
                                     void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
               : "=r"(value[0])
               : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_x2_trans(void const *const smem_ptr,
                                     void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
  asm volatile(
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
      : "=r"(value[0]), "=r"(value[1])
      : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_ldmatrix_x4_trans(void const *const smem_ptr,
                                     void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
  asm volatile(
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
      : "r"(smem_int_ptr));
}

TL_DEVICE void ptx_stmatrix_m8n8_x1(void const *const smem_ptr,
                                    const int32_t &value0) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(
                   smem_int_ptr),
               "r"(value0));
}

TL_DEVICE void ptx_stmatrix_m8n8_x2(void const *const smem_ptr,
                                    const int32_t &value0,
                                    const int32_t &value1) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(
          smem_int_ptr),
      "r"(value0), "r"(value1));
}

TL_DEVICE void ptx_stmatrix_m8n8_x4(void const *const smem_ptr,
                                    const int32_t &value0,
                                    const int32_t &value1,
                                    const int32_t &value2,
                                    const int32_t &value3) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::
          "r"(smem_int_ptr),
      "r"(value0), "r"(value1), "r"(value2), "r"(value3));
}

TL_DEVICE void ptx_stmatrix_m8n8_x1_trans(void const *const smem_ptr,
                                          const int32_t &value0) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"(
          smem_int_ptr),
      "r"(value0));
}

TL_DEVICE void ptx_stmatrix_m8n8_x2_trans(void const *const smem_ptr,
                                          const int32_t &value0,
                                          const int32_t &value1) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(
          smem_int_ptr),
      "r"(value0), "r"(value1));
}

TL_DEVICE void ptx_stmatrix_m8n8_x4_trans(void const *const smem_ptr,
                                          const int32_t &value0,
                                          const int32_t &value1,
                                          const int32_t &value2,
                                          const int32_t &value3) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, "
               "%3, %4};\n" ::"r"(smem_int_ptr),
               "r"(value0), "r"(value1), "r"(value2), "r"(value3));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000) &&                       \
    (defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM100_F))

TL_DEVICE void ptx_stmatrix_m16n8_x1_trans(void const *const smem_ptr,
                                           const int32_t &value0) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n" ::"r"(
          smem_int_ptr),
      "r"(value0));
}

TL_DEVICE void ptx_stmatrix_m16n8_x2_trans(void const *const smem_ptr,
                                           const int32_t &value0,
                                           const int32_t &value1) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n" ::"r"(
          smem_int_ptr),
      "r"(value0), "r"(value1));
}

TL_DEVICE void ptx_stmatrix_m16n8_x4_trans(void const *const smem_ptr,
                                           const int32_t &value0,
                                           const int32_t &value1,
                                           const int32_t &value2,
                                           const int32_t &value3) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, "
      "%4};\n" ::"r"(smem_int_ptr),
      "r"(value0), "r"(value1), "r"(value2), "r"(value3));
}

#endif

TL_DEVICE void ptx_stmatrix_x1(void const *const smem_ptr,
                               const int32_t &value0) {
  ptx_stmatrix_m8n8_x1(smem_ptr, value0);
}

TL_DEVICE void ptx_stmatrix_x2(void const *const smem_ptr,
                               const int32_t &value0, const int32_t &value1) {
  ptx_stmatrix_m8n8_x2(smem_ptr, value0, value1);
}

TL_DEVICE void ptx_stmatrix_x4(void const *const smem_ptr,
                               const int32_t &value0, const int32_t &value1,
                               const int32_t &value2, const int32_t &value3) {
  ptx_stmatrix_m8n8_x4(smem_ptr, value0, value1, value2, value3);
}

TL_DEVICE void ptx_stmatrix_x1_trans(void const *const smem_ptr,
                                     const int32_t &value0) {
  ptx_stmatrix_m8n8_x1_trans(smem_ptr, value0);
}

TL_DEVICE void ptx_stmatrix_x2_trans(void const *const smem_ptr,
                                     const int32_t &value0,
                                     const int32_t &value1) {
  ptx_stmatrix_m8n8_x2_trans(smem_ptr, value0, value1);
}

TL_DEVICE void ptx_stmatrix_x4_trans(void const *const smem_ptr,
                                     const int32_t &value0,
                                     const int32_t &value1,
                                     const int32_t &value2,
                                     const int32_t &value3) {
  ptx_stmatrix_m8n8_x4_trans(smem_ptr, value0, value1, value2, value3);
}

// Sub-byte ldmatrix variants (SM100/SM120 family, CUDA 12.8+): each m8n16
// source row is one 16-byte unit holding 16 packed 4-bit (b4x16_p64: 8B
// payload + 8B padding) or 6-bit (b6x16_p32: 12B payload + 4B padding)
// elements; the instruction zero-extends every element into an 8-bit
// register container (value at bits[3:0] / bits[5:0]). No trans forms
// exist. Byte footprint per matrix equals the classic m8n8.b16 form, so
// lane addressing is shared with the 8-bit path.
#define TL_DEFINE_PTX_LDMATRIX_SUB_BYTE(variant, ptx_suffix)                   \
  TL_DEVICE void ptx_ldmatrix_##variant##_x1(void const *const smem_ptr,       \
                                             void *const local_ptr) {          \
    uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);                        \
    int32_t *value = reinterpret_cast<int32_t *>(local_ptr);                   \
    asm volatile("ldmatrix.sync.aligned.m8n16.x1.shared.b8x16." ptx_suffix     \
                 " {%0}, [%1];\n"                                              \
                 : "=r"(value[0])                                              \
                 : "r"(smem_int_ptr));                                         \
  }                                                                            \
  TL_DEVICE void ptx_ldmatrix_##variant##_x2(void const *const smem_ptr,       \
                                             void *const local_ptr) {          \
    uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);                        \
    int32_t *value = reinterpret_cast<int32_t *>(local_ptr);                   \
    asm volatile("ldmatrix.sync.aligned.m8n16.x2.shared.b8x16." ptx_suffix     \
                 " {%0, %1}, [%2];\n"                                          \
                 : "=r"(value[0]), "=r"(value[1])                              \
                 : "r"(smem_int_ptr));                                         \
  }                                                                            \
  TL_DEVICE void ptx_ldmatrix_##variant##_x4(void const *const smem_ptr,       \
                                             void *const local_ptr) {          \
    uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);                        \
    int32_t *value = reinterpret_cast<int32_t *>(local_ptr);                   \
    asm volatile("ldmatrix.sync.aligned.m8n16.x4.shared.b8x16." ptx_suffix     \
                 " {%0, %1, %2, %3}, [%4];\n"                                  \
                 : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]),             \
                   "=r"(value[3])                                              \
                 : "r"(smem_int_ptr));                                         \
  }

TL_DEFINE_PTX_LDMATRIX_SUB_BYTE(su4, "b4x16_p64")
TL_DEFINE_PTX_LDMATRIX_SUB_BYTE(su6, "b6x16_p32")

#undef TL_DEFINE_PTX_LDMATRIX_SUB_BYTE

// Shift zero-extended e2m1 containers (bits[3:0]) up to the bits[5:2]
// position kind::mxf8f6f4 expects. Safe as a whole-word shift: b4x16_p64
// zero-extension guarantees bits[7:4] of every byte are zero, so nothing
// crosses byte boundaries. fp6 containers need no shift (bits[5:0]).
TL_DEVICE void fp4_e2m1_container_shift(void *const local_ptr, int num_words) {
  uint32_t *words = reinterpret_cast<uint32_t *>(local_ptr);
#pragma unroll
  for (int i = 0; i < num_words; ++i) {
    words[i] <<= 2;
  }
}

} // namespace tl
