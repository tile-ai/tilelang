#ifndef TILELANG_CMAKE_WINDOWS_ARM64_CLANG_COMPAT_H_
#define TILELANG_CMAKE_WINDOWS_ARM64_CLANG_COMPAT_H_

#if defined(_WIN32) && defined(_M_ARM64) && defined(__clang__)
#include <arm_acle.h>

#if __clang_major__ < 20
#include <intrin.h>

static inline void *TileLangInterlockedCompareExchangePointerAcquire(
    void *volatile *destination, void *exchange, void *comparand) {
  return reinterpret_cast<void *>(_InterlockedCompareExchange64_acq(
      reinterpret_cast<volatile __int64 *>(destination),
      reinterpret_cast<__int64>(exchange),
      reinterpret_cast<__int64>(comparand)));
}

// Clang 19 lacks the pointer-form acquire intrinsic used by the Windows SDK.
#define _InterlockedCompareExchangePointer_acq                                 \
  TileLangInterlockedCompareExchangePointerAcquire
#endif

// TVM's vendored compiler-rt fp16 helper also declares __clz. Load ARM ACLE
// first, then rename the vendored helper and its uses to avoid a collision.
#define __clz tvm_compiler_rt_clz
#endif

#endif // TILELANG_CMAKE_WINDOWS_ARM64_CLANG_COMPAT_H_
