#pragma once

#include "mma_sm80_sparse.hpp" // for SM80::MMA::SparseSel

#if (__CUDACC_VER_MAJOR__ > 12) ||                                             \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4)
#define CUTE_ARCH_SPARSE_MMA_SM89_SUPPORTED
#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
#if defined(CUTE_ARCH_SPARSE_MMA_SM89_SUPPORTED)
#define CUTE_ARCH_SPARSE_MMA_SM89_ENABLED
#endif
#endif

namespace SM89 {
namespace MMA {

using SM80::MMA::SparseSel;

namespace SPARSE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM89_16x8x64_F32E4M3E4M3F32_TN {
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
    static_assert(spsel == SparseSel::Zero,
                  "SM89 fp8 sparse mma only supports SparseSel::Zero");
#if defined(CUTE_ARCH_SPARSE_MMA_SM89_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM89_16x8x64_F32E4M3E4M3F32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM89_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM89_16x8x64_F32E4M3E5M2F32_TN {
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
    static_assert(spsel == SparseSel::Zero,
                  "SM89 fp8 sparse mma only supports SparseSel::Zero");
#if defined(CUTE_ARCH_SPARSE_MMA_SM89_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e4m3.e5m2.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e4m3.e5m2.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM89_16x8x64_F32E4M3E5M2F32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM89_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM89_16x8x64_F32E5M2E4M3F32_TN {
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
    static_assert(spsel == SparseSel::Zero,
                  "SM89 fp8 sparse mma only supports SparseSel::Zero");
#if defined(CUTE_ARCH_SPARSE_MMA_SM89_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e5m2.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM89_16x8x64_F32E5M2E4M3F32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM89_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <SparseSel spsel = SparseSel::Zero>
struct SM89_16x8x64_F32E5M2E5M2F32_TN {
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
    static_assert(spsel == SparseSel::Zero,
                  "SM89 fp8 sparse mma only supports SparseSel::Zero");
#if defined(CUTE_ARCH_SPARSE_MMA_SM89_ENABLED)
#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))
    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k64.row.col.f32."
                 "e5m2.e5m2.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
#else
    asm volatile("mma.sp.sync.aligned.m16n8k64.row.col.f32.e5m2.e5m2.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, "
                 "{%12,%13,%14,%15}, %16, 0x0;\n"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                   "r"(e));
#endif
#else
    CUTE_INVALID_CONTROL_PATH("SM89_16x8x64_F32E5M2E5M2F32_TN requires "
                              "CUTE_ARCH_SPARSE_MMA_SM89_ENABLED");
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace SPARSE
} // namespace MMA
} // namespace SM89
