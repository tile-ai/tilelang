#pragma once

#include "common.h"

// Memory semantic and scope enums
enum class Semantic { WEAK, VOLATILE, ACQUIRE, RELEASE, RELAXED };
enum class Scope { CTA, GPU, SYS };

namespace tl {

#ifndef TL_ALWAYS_FALSE_V_DEFINED
#define TL_ALWAYS_FALSE_V_DEFINED
template <class> inline constexpr bool always_false_v = false;
#endif

// Type trait to detect bfloat16 types
template <typename T> struct is_bfloat16 : std::false_type {};

#ifdef __CUDA_BF16_TYPES_EXIST__
template <> struct is_bfloat16<__nv_bfloat16> : std::true_type {};
#endif

} // namespace tl

// Detect cutlass bfloat16_t
namespace cutlass {
struct bfloat16_t;
}
template <> struct tl::is_bfloat16<cutlass::bfloat16_t> : std::true_type {};

template <typename T>
inline constexpr bool is_bfloat16_v = tl::is_bfloat16<T>::value;

// Fallback template for unsupported configurations
template <Semantic semantic, Scope scope, bool na> struct StImpl {
  template <typename T> TL_DEVICE static void execute(T *ptr, T value) {
    static_assert(tl::always_false_v<T>, "tl::st: unsupported configuration. ");
  }
};

template <Semantic semantic, Scope scope, bool nc, bool na> struct LdImpl {
  template <typename T> TL_DEVICE static void execute(const T *ptr, T &value) {
    static_assert(tl::always_false_v<T>, "tl::ld: unsupported configuration. ");
  }
};

// Macro to define implementation with generic type T
#define TL_ST_IMPL(SEM, SCOPE, NA, SEM_LIT, SCOPE_LIT, NA_LIT)                 \
  template <> struct StImpl<Semantic::SEM, Scope::SCOPE, NA> {                 \
    template <typename T> TL_DEVICE static void execute(T *ptr, T value) {     \
      if constexpr (sizeof(T) == 2) {                                          \
        if constexpr (is_bfloat16_v<T>) {                                      \
          uint16_t value_bits = *reinterpret_cast<uint16_t *>(&value);         \
          asm volatile("st" SEM_LIT SCOPE_LIT NA_LIT                           \
                       ".b16 [%0], %1;" ::"l"(ptr),                            \
                       "h"(value_bits)                                         \
                       : "memory");                                            \
        } else {                                                               \
          asm volatile("st" SEM_LIT SCOPE_LIT NA_LIT                           \
                       ".b16 [%0], %1;" ::"l"(ptr),                            \
                       "h"(value)                                              \
                       : "memory");                                            \
        }                                                                      \
      } else if constexpr (sizeof(T) == 4) {                                   \
        if constexpr (std::is_floating_point_v<T>) {                           \
          asm volatile("st" SEM_LIT SCOPE_LIT NA_LIT                           \
                       ".b32 [%0], %1;" ::"l"(ptr),                            \
                       "f"(value)                                              \
                       : "memory");                                            \
        } else {                                                               \
          asm volatile("st" SEM_LIT SCOPE_LIT NA_LIT                           \
                       ".b32 [%0], %1;" ::"l"(ptr),                            \
                       "r"(value)                                              \
                       : "memory");                                            \
        }                                                                      \
      } else if constexpr (sizeof(T) == 8) {                                   \
        if constexpr (std::is_floating_point_v<T>) {                           \
          asm volatile("st" SEM_LIT SCOPE_LIT NA_LIT                           \
                       ".b64 [%0], %1;" ::"l"(ptr),                            \
                       "d"(value)                                              \
                       : "memory");                                            \
        } else {                                                               \
          asm volatile("st" SEM_LIT SCOPE_LIT NA_LIT                           \
                       ".b64 [%0], %1;" ::"l"(ptr),                            \
                       "l"(value)                                              \
                       : "memory");                                            \
        }                                                                      \
      } else if constexpr (sizeof(T) == 16) {                                  \
        asm volatile("st" SEM_LIT SCOPE_LIT NA_LIT                             \
                     ".v4.s32 [%0], {%1, %2, %3, %4};" ::"l"(ptr),             \
                     "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)    \
                     : "memory");                                              \
      }                                                                        \
    }                                                                          \
  };

// Macro to define implementation of tl::ld with generic type T
#define TL_LD_IMPL(SEM, SCOPE, NC, NA, SEM_LIT, SCOPE_LIT, NC_LIT, NA_LIT)     \
  template <> struct LdImpl<Semantic::SEM, Scope::SCOPE, NC, NA> {             \
    template <typename T>                                                      \
    TL_DEVICE static void execute(const T *ptr, T &value) {                    \
      if constexpr (sizeof(T) == 2) {                                          \
        if constexpr (is_bfloat16_v<T>) {                                      \
          uint16_t value_bits;                                                 \
          asm volatile("ld" SEM_LIT SCOPE_LIT NC_LIT NA_LIT ".b16 %0, [%1];"   \
                       : "=h"(value_bits)                                      \
                       : "l"(ptr)                                              \
                       : "memory");                                            \
          value = *reinterpret_cast<T *>(&value_bits);                         \
        } else {                                                               \
          asm volatile("ld" SEM_LIT SCOPE_LIT NC_LIT NA_LIT ".b16 %0, [%1];"   \
                       : "=h"(value)                                           \
                       : "l"(ptr)                                              \
                       : "memory");                                            \
        }                                                                      \
      } else if constexpr (sizeof(T) == 4) {                                   \
        if constexpr (std::is_floating_point_v<T>) {                           \
          asm volatile("ld" SEM_LIT SCOPE_LIT NC_LIT NA_LIT ".b32 %0, [%1];"   \
                       : "=f"(value)                                           \
                       : "l"(ptr)                                              \
                       : "memory");                                            \
        } else {                                                               \
          asm volatile("ld" SEM_LIT SCOPE_LIT NC_LIT NA_LIT ".b32 %0, [%1];"   \
                       : "=r"(value)                                           \
                       : "l"(ptr)                                              \
                       : "memory");                                            \
        }                                                                      \
      } else if constexpr (sizeof(T) == 8) {                                   \
        if constexpr (std::is_floating_point_v<T>) {                           \
          asm volatile("ld" SEM_LIT SCOPE_LIT NC_LIT NA_LIT ".b64 %0, [%1];"   \
                       : "=d"(value)                                           \
                       : "l"(ptr)                                              \
                       : "memory");                                            \
        } else {                                                               \
          asm volatile("ld" SEM_LIT SCOPE_LIT NC_LIT NA_LIT ".b64 %0, [%1];"   \
                       : "=l"(value)                                           \
                       : "l"(ptr)                                              \
                       : "memory");                                            \
        }                                                                      \
      } else if constexpr (sizeof(T) == 16) {                                  \
        asm volatile("ld" SEM_LIT SCOPE_LIT NC_LIT NA_LIT                      \
                     ".v4.s32 {%0, %1, %2, %3}, [%4];"                         \
                     : "=r"(value.x), "=r"(value.y), "=r"(value.z),            \
                       "=r"(value.w)                                           \
                     : "l"(ptr)                                                \
                     : "memory");                                              \
      }                                                                        \
    }                                                                          \
  };

// Register all combinations of arguments for tl::st in need here
// WEAK (always .global)
TL_ST_IMPL(WEAK, CTA, false, ".weak", ".global", "")
TL_ST_IMPL(WEAK, GPU, false, ".weak", ".global", "")
TL_ST_IMPL(WEAK, GPU, true, ".weak", ".global", ".L1::no_allocate")
TL_ST_IMPL(WEAK, SYS, false, ".weak", ".global", "")
TL_ST_IMPL(WEAK, SYS, true, ".weak", ".global", ".L1::no_allocate")

// VOLATILE (always .global, no na)
TL_ST_IMPL(VOLATILE, CTA, false, ".volatile", ".global", "")
TL_ST_IMPL(VOLATILE, GPU, false, ".volatile", ".global", "")
TL_ST_IMPL(VOLATILE, SYS, false, ".volatile", ".global", "")

// RELAXED (scope-aware)
TL_ST_IMPL(RELAXED, CTA, false, ".relaxed", ".cta", "")
TL_ST_IMPL(RELAXED, GPU, false, ".relaxed", ".gpu.global", "")
TL_ST_IMPL(RELAXED, GPU, true, ".relaxed", ".gpu.global", ".L1::no_allocate")
TL_ST_IMPL(RELAXED, SYS, false, ".relaxed", ".sys.global", "")
TL_ST_IMPL(RELAXED, SYS, true, ".relaxed", ".sys.global", ".L1::no_allocate")

// RELEASE (scope-aware)
TL_ST_IMPL(RELEASE, CTA, false, ".release", ".cta", "")
TL_ST_IMPL(RELEASE, GPU, false, ".release", ".gpu.global", "")
TL_ST_IMPL(RELEASE, GPU, true, ".release", ".gpu.global", ".L1::no_allocate")
TL_ST_IMPL(RELEASE, SYS, false, ".release", ".sys.global", "")
TL_ST_IMPL(RELEASE, SYS, true, ".release", ".sys.global", ".L1::no_allocate")

// Register all combinations of arguments for tl::ld in need here
// nc (must with no scope and semantic)
TL_LD_IMPL(WEAK, CTA, true, false, "", ".global", ".nc", "")
TL_LD_IMPL(WEAK, GPU, true, false, "", ".global", ".nc", "")
TL_LD_IMPL(WEAK, SYS, true, false, "", ".global", ".nc", "")
TL_LD_IMPL(WEAK, GPU, true, true, "", ".global", ".nc", ".L1::no_allocate")
TL_LD_IMPL(WEAK, SYS, true, true, "", ".global", ".nc", ".L1::no_allocate")

// WEAK
TL_LD_IMPL(WEAK, CTA, false, false, ".weak", ".cta", "", "")
TL_LD_IMPL(WEAK, GPU, false, false, ".weak", ".gpu.global", "", "")
TL_LD_IMPL(WEAK, SYS, false, false, ".weak", ".sys.global", "", "")
TL_LD_IMPL(WEAK, GPU, false, true, ".weak", ".gpu.global", "",
           ".L1::no_allocate")
TL_LD_IMPL(WEAK, SYS, false, true, ".weak", ".sys.global", "",
           ".L1::no_allocate")

// VOLATILE (always .global, no na)
TL_LD_IMPL(VOLATILE, CTA, false, false, ".volatile", ".global", "", "")
TL_LD_IMPL(VOLATILE, GPU, false, false, ".volatile", ".global", "", "")
TL_LD_IMPL(VOLATILE, SYS, false, false, ".volatile", ".global", "", "")

// RELAXED (scope-aware)
TL_LD_IMPL(RELAXED, CTA, false, false, ".relaxed", ".cta", "", "")
TL_LD_IMPL(RELAXED, GPU, false, false, ".relaxed", ".gpu.global", "", "")
TL_LD_IMPL(RELAXED, SYS, false, false, ".relaxed", ".sys.global", "", "")
TL_LD_IMPL(RELAXED, GPU, false, true, ".relaxed", ".gpu.global", "",
           ".L1::no_allocate")
TL_LD_IMPL(RELAXED, SYS, false, true, ".relaxed", ".sys.global", "",
           ".L1::no_allocate")

// ACQUIRE (scope-aware)
TL_LD_IMPL(ACQUIRE, CTA, false, false, ".acquire", ".cta", "", "")
TL_LD_IMPL(ACQUIRE, GPU, false, false, ".acquire", ".gpu.global", "", "")
TL_LD_IMPL(ACQUIRE, SYS, false, false, ".acquire", ".sys.global", "", "")
TL_LD_IMPL(ACQUIRE, GPU, false, true, ".acquire", ".gpu.global", "",
           ".L1::no_allocate")
TL_LD_IMPL(ACQUIRE, SYS, false, true, ".acquire", ".sys.global", "",
           ".L1::no_allocate")

#undef TL_ST_IMPL
#undef TL_LD_IMPL

namespace tl {

// Public interface
template <Semantic semantic, Scope scope, bool na, typename P, typename T>
TL_DEVICE void st(P ptr, T value) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8 ||
                    sizeof(T) == 16,
                "tl::st: T must be 2, 4, 8, or 16 bytes");
  static_assert(std::is_pointer_v<P> || std::is_same_v<P, uint64_t>,
                "tl::st: P must be a pointer or uint64_t");
  static_assert(semantic == Semantic::WEAK || semantic == Semantic::RELAXED ||
                    semantic == Semantic::RELEASE ||
                    semantic == Semantic::VOLATILE,
                "tl::st: semantic must be WEAK, VOLATILE, RELAXED, or RELEASE");

  T *ptr_ = reinterpret_cast<T *>(ptr);
  StImpl<semantic, scope, na>::execute(ptr_, value);
}

template <Semantic semantic, Scope scope, bool nc, bool na, typename P,
          typename T>
TL_DEVICE void ld(const P ptr, T &value) {
  static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8 ||
                    sizeof(T) == 16,
                "tl::ld: T must be 2, 4, 8, or 16 bytes");
  static_assert(std::is_pointer_v<P> || std::is_same_v<P, uint64_t>,
                "tl::ld: P must be a pointer or uint64_t");
  static_assert(semantic == Semantic::WEAK || semantic == Semantic::RELAXED ||
                    semantic == Semantic::ACQUIRE ||
                    semantic == Semantic::VOLATILE,
                "tl::ld: semantic must be WEAK, RELAXED, ACQUIRE, or VOLATILE");

  const T *ptr_ = reinterpret_cast<const T *>(ptr);
  LdImpl<semantic, scope, nc, na>::execute(ptr_, value);
}

// todo: support "ld.global.nc.L1::no_allocate.L2::256B"

} // namespace tl
