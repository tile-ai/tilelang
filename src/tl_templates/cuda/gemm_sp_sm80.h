#include <cute/arch/mma_sm80.hpp>

// TODO: remove
#define CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED

namespace cute {

enum class SparseSel {
  Zero = 0,
  One = 1,
  Two = 2,
  Three = 3,
};

// ref:
// https://github.com/botbw/cutlass/blob/ad7b2f5e84fcfa124cb02b91d5bd26d238c0459e/include/cutlass/arch/mma_sparse_sm80.h#L70
template <SparseSel spsel = SparseSel::Zero>
struct SM80_SPARSE_16x8x32_F16F16F16F16_TN {
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
#if defined(CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED)

#if ((__CUDACC_VER_MAJOR__ > 12) ||                                            \
     (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 5))

    asm volatile("mma.sp::ordered_metadata.sync.aligned.m16n8k32.row.col.f16."
                 "f16.f16.f16 {%0,%1}, "
                 "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, %13;\n"
                 : "=r"(d0), "=r"(d1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "r"(b2), "r"(b3), "r"(c0), "r"(c1), "r"(e),
                   "r"(int32_t(spsel)));
#else
    asm volatile(
        "mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, "
        "{%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, %13;\n"
        : "=r"(d0), "=r"(d1)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2),
          "r"(b3), "r"(c0), "r"(c1), "r"(e), "r"(int32_t(spsel)));
#endif
#else
    CUTE_INVALID_CONTROL_PATH(
        "TileLang: Attempted to use SM80_SPARSE_16x8x32_F16F16F16F16_TN "
        "without CUTLASS_ARCH_SPARSE_MMA_SM80_ENABLED");
#endif
  }
};
} // namespace cute