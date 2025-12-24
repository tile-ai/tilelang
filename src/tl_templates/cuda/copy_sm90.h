#pragma once

#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include "barrier.h"
#include "common.h"

namespace tl {
enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};

template <typename BarrierType = uint64_t>
TL_DEVICE void tma_load(void *smem_ptr, void const *gmem_ptr,
                        BarrierType &smem_mbar, uint32_t size) {
  uint32_t smem_int_mbar =
      smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3]; \n" ::"r"(smem_int_ptr),
               "l"((void const *)gmem_ptr), "r"(size), "r"(smem_int_mbar)
               :);
}

TL_DEVICE void tma_load_multicast(void *smem_ptr, void *gmem_ptr,
                                  uint64_t &smem_mbar, uint32_t size,
                                  uint16_t mask) {
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes."
      "multicast::cluster [%0], [%1], %2, [%3], %4; \n" ::"r"(smem_int_ptr),
      "l"(gmem_ptr), "r"(size), "r"(smem_int_mbar), "h"(mask)
      :);
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3}], [%2], %4;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4}], [%2], %5;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5}], [%2], %6;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
               : "memory");
}
template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2,
                        int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2,
                        int32_t const &crd3, int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4),
                 "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL,
          typename BarrierType = uint64_t>
TL_DEVICE void
tma_load_im2col(const CUtensorMap &descriptor, BarrierType &smem_mbar,
                void const *const smem_ptr, int32_t const &coord_c,
                int32_t const &coord_w, int32_t const &coord_h,
                int32_t const &coord_n, uint16_t const &offset_w,
                uint16_t const &offset_h) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar =
      smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier:"
               ":complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8}, %9;"
               :
               : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                 "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n),
                 "h"(offset_w), "h"(offset_h), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(void *gmem_ptr, void *smem_ptr, uint32_t size) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group"
               ".L2::cache_hint [%0], [%1], %2, %3;"
               :
               : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(size), "l"(cache_hint)
               :);
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group "
               ".L2::cache_hint [%0, {%2}], [%1], %3;"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0),
                 "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
               ".L2::cache_hint [%0, {%2, %3}], [%1], %4;"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                 "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1, int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
               ".L2::cache_hint [%0, {%2, %3, %4}], [%1], %5;"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                 "r"(crd2), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1, int32_t const &crd2,
                         int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
               ".L2::cache_hint [%0, {%2, %3, %4, %5}], [%1], %6;"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                 "r"(crd2), "r"(crd3), "l"(cache_hint)
               : "memory");
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1, int32_t const &crd2,
                         int32_t const &crd3, int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
               ".L2::cache_hint [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
               :
               : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                 "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
               : "memory");
}

TL_DEVICE void tma_store_add(float *const smem_ptr, float *gmem_ptr,
                             int32_t const &store_bytes) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 "
               "[%0], [%1], %2;\n"
               :
               : "l"(gmem_ptr), "r"(smem_int_ptr), "r"(store_bytes)
               : "memory");
}

TL_DEVICE void prefetch_tma_descriptor(const CUtensorMap &descriptor) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
}

} // namespace tl
