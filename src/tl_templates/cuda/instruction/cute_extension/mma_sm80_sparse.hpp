// NOTE: CUTLASS didn't implement this for sm8x
#pragma once

#include <cute/arch/mma.hpp>
#include <cute/config.hpp>

// Config
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define CUTE_ARCH_SPARSE_MMA_SM80_ENABLED
#endif

namespace SM80 {
namespace MMA {

enum class SparseSel : int { Zero = 0, One = 1 };

namespace SPARSE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM80_16x8x16_F16F16F16F16_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[2];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1,
                                   uint32_t const &a0, uint32_t const &a1,
                                   uint32_t const &b0, uint32_t const &b1,
                                   uint32_t const &c0, uint32_t const &c1,
                                   uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16."
                 "f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, %9;\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(e),
                   "n"(int32_t(spsel)));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, %9;\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(e),
                   "n"(int32_t(spsel)));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x16_F16F16F16F16_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM80_16x8x32_F16F16F16F16_TN {
  using DRegisters = uint32_t[2];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[2];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(uint32_t &d0, uint32_t &d1, uint32_t const &a0, uint32_t const &a1,
      uint32_t const &a2, uint32_t const &a3, uint32_t const &b0,
      uint32_t const &b1, uint32_t const &b2, uint32_t const &b3,
      uint32_t const &c0, uint32_t const &c1, uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16."
                 "f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, %13;\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(c0), "r"(c1), "r"(e),
                   "n"(int32_t(spsel)));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, %13;\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(c0), "r"(c1), "r"(e),
                   "n"(int32_t(spsel)));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x32_F16F16F16F16_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM80_16x8x16_F32F16F16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(float &d0, float &d1, float &d2, float &d3,
                                   uint32_t const &a0, uint32_t const &a1,
                                   uint32_t const &b0, uint32_t const &b1,
                                   float const &c0, float const &c1,
                                   float const &c2, float const &c3,
                                   uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32."
                 "f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, %13;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
                   "f"(c2), "f"(c3), "r"(e), "n"(int32_t(spsel)));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, %13;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
                   "f"(c2), "f"(c3), "r"(e), "n"(int32_t(spsel)));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x16_F32F16F16F32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM80_16x8x32_F32F16F16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(float &d0, float &d1, float &d2, float &d3, uint32_t const &a0,
      uint32_t const &a1, uint32_t const &a2, uint32_t const &a3,
      uint32_t const &b0, uint32_t const &b1, uint32_t const &b2,
      uint32_t const &b3, float const &c0, float const &c1, float const &c2,
      float const &c3, uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32."
                 "f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, %17;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(e),
                   "n"(int32_t(spsel)));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, %17;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(e),
                   "n"(int32_t(spsel)));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x32_F32F16F16F32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM80_16x8x16_F32BF16BF16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(float &d0, float &d1, float &d2, float &d3,
                                   uint32_t const &a0, uint32_t const &a1,
                                   uint32_t const &b0, uint32_t const &b1,
                                   float const &c0, float const &c1,
                                   float const &c2, float const &c3,
                                   uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32."
                 "bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, %13;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
                   "f"(c2), "f"(c3), "r"(e), "n"(int32_t(spsel)));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, %13;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
                   "f"(c2), "f"(c3), "r"(e), "n"(int32_t(spsel)));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x16_F32BF16BF16F32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM80_16x8x32_F32BF16BF16F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(float &d0, float &d1, float &d2, float &d3, uint32_t const &a0,
      uint32_t const &a1, uint32_t const &a2, uint32_t const &a3,
      uint32_t const &b0, uint32_t const &b1, uint32_t const &b2,
      uint32_t const &b3, float const &c0, float const &c1, float const &c2,
      float const &c3, uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f32."
                 "bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, %17;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(e),
                   "n"(int32_t(spsel)));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, %17;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(e),
                   "n"(int32_t(spsel)));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x32_F32BF16BF16F32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM80_16x8x8_F32TF32TF32F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = float[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(float &d0, float &d1, float &d2, float &d3,
                                   uint32_t const &a0, uint32_t const &a1,
                                   uint32_t const &b0, uint32_t const &b1,
                                   float const &c0, float const &c1,
                                   float const &c2, float const &c3,
                                   uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k8.row.col.f32."
                 "tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, %13;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
                   "f"(c2), "f"(c3), "r"(e), "n"(int32_t(spsel)));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, %13;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
                   "f"(c2), "f"(c3), "r"(e), "n"(int32_t(spsel)));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x8_F32TF32TF32F32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM80_16x8x16_F32TF32TF32F32_TN {
  using DRegisters = float[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = float[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void
  fma(float &d0, float &d1, float &d2, float &d3, uint32_t const &a0,
      uint32_t const &a1, uint32_t const &a2, uint32_t const &a3,
      uint32_t const &b0, uint32_t const &b1, uint32_t const &b2,
      uint32_t const &b3, float const &c0, float const &c1, float const &c2,
      float const &c3, uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f32."
                 "tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, %17;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(e),
                   "n"(int32_t(spsel)));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, %17;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3), "r"(e),
                   "n"(int32_t(spsel)));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x16_F32TF32TF32F32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero> struct SM80_16x8x32_S32S8S8S32_TN {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1, uint32_t &d2,
                                   uint32_t &d3, uint32_t const &a0,
                                   uint32_t const &a1, uint32_t const &b0,
                                   uint32_t const &b1, uint32_t const &c0,
                                   uint32_t const &c1, uint32_t const &c2,
                                   uint32_t const &c3, uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2),
          "r"(c3), "r"(e));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1),
                   "r"(c2), "r"(c3), "r"(e));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x32_S32S8S8S32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

template <> struct SM80_16x8x32_S32S8S8S32_TN<SparseSel::One> {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1, uint32_t &d2,
                                   uint32_t &d3, uint32_t const &a0,
                                   uint32_t const &a1, uint32_t const &b0,
                                   uint32_t const &b1, uint32_t const &c0,
                                   uint32_t const &c1, uint32_t const &c2,
                                   uint32_t const &c3, uint32_t const &e) {
    CUTE_INVALID_CONTROL_PATH(
        "SM80_16x8x32_S32S8S8S32_TN with SparseSel::One is invalid");
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero> struct SM80_16x8x64_S32S8S8S32_TN {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1, uint32_t &d2,
                                   uint32_t &d3, uint32_t const &a0,
                                   uint32_t const &a1, uint32_t const &a2,
                                   uint32_t const &a3, uint32_t const &b0,
                                   uint32_t const &b1, uint32_t const &b2,
                                   uint32_t const &b3, uint32_t const &c0,
                                   uint32_t const &c1, uint32_t const &c2,
                                   uint32_t const &c3, uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
        "%16, 0x0;\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(e));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                   "r"(e));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x64_S32S8S8S32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

template <> struct SM80_16x8x64_S32S8S8S32_TN<SparseSel::One> {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1, uint32_t &d2,
                                   uint32_t &d3, uint32_t const &a0,
                                   uint32_t const &a1, uint32_t const &a2,
                                   uint32_t const &a3, uint32_t const &b0,
                                   uint32_t const &b1, uint32_t const &b2,
                                   uint32_t const &b3, uint32_t const &c0,
                                   uint32_t const &c1, uint32_t const &c2,
                                   uint32_t const &c3, uint32_t const &e) {
    CUTE_INVALID_CONTROL_PATH(
        "SM80_16x8x64_S32S8S8S32_TN with SparseSel::One is invalid");
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero> struct SM80_16x8x32_S32U8U8S32_TN {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1, uint32_t &d2,
                                   uint32_t &d3, uint32_t const &a0,
                                   uint32_t const &a1, uint32_t const &b0,
                                   uint32_t const &b1, uint32_t const &c0,
                                   uint32_t const &c1, uint32_t const &c2,
                                   uint32_t const &c3, uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1), "r"(c2),
          "r"(c3), "r"(e));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.s32.u8.u8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(c1),
                   "r"(c2), "r"(c3), "r"(e));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x32_S32U8U8S32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

template <> struct SM80_16x8x32_S32U8U8S32_TN<SparseSel::One> {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[2];
  using BRegisters = uint32_t[2];
  using CRegisters = uint32_t[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1, uint32_t &d2,
                                   uint32_t &d3, uint32_t const &a0,
                                   uint32_t const &a1, uint32_t const &b0,
                                   uint32_t const &b1, uint32_t const &c0,
                                   uint32_t const &c1, uint32_t const &c2,
                                   uint32_t const &c3, uint32_t const &e) {
    CUTE_INVALID_CONTROL_PATH(
        "SM80_16x8x32_S32U8U8S32_TN with SparseSel::One is invalid");
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero> struct SM80_16x8x64_S32U8U8S32_TN {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1, uint32_t &d2,
                                   uint32_t &d3, uint32_t const &a0,
                                   uint32_t const &a1, uint32_t const &a2,
                                   uint32_t const &a3, uint32_t const &b0,
                                   uint32_t const &b1, uint32_t const &b2,
                                   uint32_t const &b3, uint32_t const &c0,
                                   uint32_t const &c1, uint32_t const &c2,
                                   uint32_t const &c3, uint32_t const &e) {
#if defined(CUTE_ARCH_SPARSE_MMA_SM80_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile(
        "mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.s32.u8.u8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, "
        "%16, 0x0;\n"
        : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(e));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.s32.u8.u8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                   "r"(e));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM80_16x8x64_S32U8U8S32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};

template <> struct SM80_16x8x64_S32U8U8S32_TN<SparseSel::One> {
  using DRegisters = uint32_t[4];
  using ARegisters = uint32_t[4];
  using BRegisters = uint32_t[4];
  using CRegisters = uint32_t[4];
  using ERegisters = uint32_t[1];

  CUTE_HOST_DEVICE static void fma(uint32_t &d0, uint32_t &d1, uint32_t &d2,
                                   uint32_t &d3, uint32_t const &a0,
                                   uint32_t const &a1, uint32_t const &a2,
                                   uint32_t const &a3, uint32_t const &b0,
                                   uint32_t const &b1, uint32_t const &b2,
                                   uint32_t const &b3, uint32_t const &c0,
                                   uint32_t const &c1, uint32_t const &c2,
                                   uint32_t const &c3, uint32_t const &e) {
    CUTE_INVALID_CONTROL_PATH(
        "SM80_16x8x64_S32U8U8S32_TN with SparseSel::One is invalid");
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace SPARSE
} // namespace MMA
} // namespace SM80
