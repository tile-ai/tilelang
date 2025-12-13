#pragma once

#include "common.h"

#ifdef __CUDA_ARCH_LIST__
#if __CUDA_ARCH_LIST__ >= 900
#include "copy_sm90.h"
#endif
#if __CUDA_ARCH_LIST__ >= 1000
#include "copy_sm100.h"
#endif
#endif

namespace tl {

TL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> TL_DEVICE void cp_async_wait() {
  if constexpr (N == 0) {
    asm volatile("cp.async.wait_all;\n" ::);
  } else {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
  }
}

template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr,
                           void const *global_ptr) {
  static_assert(N == 16 || N == 8 || N == 4);
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  if constexpr (N == 16) {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
#else
        "cp.async.cg.shared.global [%0], [%1], %2;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N));
  } else {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;"
#else
        "cp.async.ca.shared.global [%0], [%1], %2;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N));
  }
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *global_ptr, bool cond) {
  static_assert(N == 16 || N == 8 || N == 4);
  int bytes = cond ? N : 0;
  unsigned int addr = smem_ptr_to_uint(smem_addr);
  if constexpr (N == 16) {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
        "cp.async.cg.shared.global [%0], [%1], %2, %3;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N), "r"(bytes));
  } else {
    asm volatile(
#if TL_ENABLE_L2_PREFETCH
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2, %3;"
#else
        "cp.async.ca.shared.global [%0], [%1], %2, %3;"
#endif
        ::"r"(addr),
        "l"((void const *)(global_ptr)), "n"(N), "r"(bytes));
  }
}

} // namespace tl
