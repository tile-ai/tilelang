#ifndef TILELANG_CMAKE_WINDOWS_ARM64_CLANG_COMPAT_H_
#define TILELANG_CMAKE_WINDOWS_ARM64_CLANG_COMPAT_H_

#if defined(_WIN32) && defined(_M_ARM64) && defined(__clang__)
#include <arm_acle.h>

// TVM's vendored compiler-rt fp16 helper also declares __clz. Load ARM ACLE
// first, then rename the vendored helper and its uses to avoid a collision.
#define __clz tvm_compiler_rt_clz
#endif

#endif // TILELANG_CMAKE_WINDOWS_ARM64_CLANG_COMPAT_H_
