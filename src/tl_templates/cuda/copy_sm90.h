#pragma once

#ifndef __CUDACC_RTC__
#include <cuda.h>
#endif

#include "common.h"

/**
 * Issue a bulk cp.async TMA load from global to shared memory and mark the
 * associated shared mbarrier transaction complete for `size` bytes.
 * @param smem_ptr Destination shared-memory pointer.
 * @param gmem_ptr Source global-memory pointer.
 * @param smem_mbar Shared mbarrier state used to complete the transaction.
 * @param size Number of bytes to transfer.
 */
/**
 * Issue a multicast-capable bulk cp.async TMA load from global to shared
 * memory and mark the associated shared mbarrier transaction complete.
 * @param smem_ptr Destination shared-memory pointer.
 * @param gmem_ptr Source global-memory pointer.
 * @param smem_mbar Shared mbarrier state used to complete the transaction.
 * @param size Number of bytes to transfer.
 * @param mask Multicast mask controlling which clusters receive the transfer.
 */
/**
 * 1D tensor-map cp.async load into shared memory and complete a shared
 * mbarrier transaction.
 * Template parameter `cache_hint` selects the cache path emitted by the
 * instruction (default EVICT_NORMAL).
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_mbar Shared mbarrier state used to complete the transaction.
 * @param smem_ptr Destination shared-memory pointer.
 * @param crd0 Coordinate for the 1D tensor index.
 */
/**
 * 2D tensor-map cp.async load into shared memory and complete a shared
 * mbarrier transaction.
 * Template parameter `cache_hint` selects the cache path emitted by the
 * instruction (default EVICT_NORMAL).
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_mbar Shared mbarrier state used to complete the transaction.
 * @param smem_ptr Destination shared-memory pointer.
 * @param crd0 First tensor coordinate.
 * @param crd1 Second tensor coordinate.
 */
/**
 * 3D tensor-map cp.async load into shared memory and complete a shared
 * mbarrier transaction.
 * Template parameter `cache_hint` selects the cache path emitted by the
 * instruction (default EVICT_NORMAL).
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_mbar Shared mbarrier state used to complete the transaction.
 * @param smem_ptr Destination shared-memory pointer.
 * @param crd0 First tensor coordinate.
 * @param crd1 Second tensor coordinate.
 * @param crd2 Third tensor coordinate.
 */
/**
 * 4D tensor-map cp.async load into shared memory and complete a shared
 * mbarrier transaction.
 * Template parameter `cache_hint` selects the cache path emitted by the
 * instruction (default EVICT_NORMAL).
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_mbar Shared mbarrier state used to complete the transaction.
 * @param smem_ptr Destination shared-memory pointer.
 * @param crd0 First tensor coordinate.
 * @param crd1 Second tensor coordinate.
 * @param crd2 Third tensor coordinate.
 * @param crd3 Fourth tensor coordinate.
 */
/**
 * 5D tensor-map cp.async load into shared memory and complete a shared
 * mbarrier transaction.
 * Template parameter `cache_hint` selects the cache path emitted by the
 * instruction (default EVICT_NORMAL).
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_mbar Shared mbarrier state used to complete the transaction.
 * @param smem_ptr Destination shared-memory pointer.
 * @param crd0 First tensor coordinate.
 * @param crd1 Second tensor coordinate.
 * @param crd2 Third tensor coordinate.
 * @param crd3 Fourth tensor coordinate.
 * @param crd4 Fifth tensor coordinate.
 */
/**
 * im2col-style 4D tensor-map cp.async load into shared memory with explicit
 * per-element offsets (used for convolution im2col access). Completes the
 * associated shared mbarrier transaction.
 * Template parameter `cache_hint` selects the cache path emitted by the
 * instruction (default EVICT_NORMAL).
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_mbar Shared mbarrier state used to complete the transaction.
 * @param smem_ptr Destination shared-memory pointer.
 * @param coord_c Channel coordinate.
 * @param coord_w Width coordinate.
 * @param coord_h Height coordinate.
 * @param coord_n Batch/sequence coordinate.
 * @param offset_w Per-element width offset (uint16).
 * @param offset_h Per-element height offset (uint16).
 */
/**
 * 1D tensor-map cp.async store from shared to global memory (CTA bulk group).
 * Template parameter `cache_hint` selects whether a cache-hinted path is used.
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_ptr Source shared-memory pointer.
 * @param crd0 Coordinate for the 1D tensor index.
 */
/**
 * 2D tensor-map cp.async store from shared to global memory (CTA bulk group).
 * Template parameter `cache_hint` selects whether a cache-hinted path is used.
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_ptr Source shared-memory pointer.
 * @param crd0 First tensor coordinate.
 * @param crd1 Second tensor coordinate.
 */
/**
 * 3D tensor-map cp.async store from shared to global memory (CTA bulk group).
 * Template parameter `cache_hint` selects whether a cache-hinted path is used.
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_ptr Source shared-memory pointer.
 * @param crd0 First tensor coordinate.
 * @param crd1 Second tensor coordinate.
 * @param crd2 Third tensor coordinate.
 */
/**
 * 4D tensor-map cp.async store from shared to global memory (CTA bulk group).
 * Template parameter `cache_hint` selects whether a cache-hinted path is used.
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_ptr Source shared-memory pointer.
 * @param crd0 First tensor coordinate.
 * @param crd1 Second tensor coordinate.
 * @param crd2 Third tensor coordinate.
 * @param crd3 Fourth tensor coordinate.
 */
/**
 * 5D tensor-map cp.async store from shared to global memory (CTA bulk group).
 * Template parameter `cache_hint` selects whether a cache-hinted path is used.
 * @param descriptor CUtensorMap describing the global tensor layout.
 * @param smem_ptr Source shared-memory pointer.
 * @param crd0 First tensor coordinate.
 * @param crd1 Second tensor coordinate.
 * @param crd2 Third tensor coordinate.
 * @param crd3 Fourth tensor coordinate.
 * @param crd4 Fifth tensor coordinate.
 */
/**
 * Prefetch the CUtensorMap descriptor into cache to hide descriptor load
 * latency.
 * @param descriptor Descriptor to prefetch.
 */
/**
 * Initialize a shared-memory mbarrier with the number of expected arriving
 * participants.
 * @param smem_barrier Shared-memory storage for mbarrier state.
 * @param arrive_count Number of arrivals expected before the barrier completes.
 */
/**
 * Attempt a parity-based try-wait on a shared mbarrier.
 * Returns 1 if the barrier is complete for the requested phase, 0 otherwise.
 * @param smem_barrier Shared-memory storage for mbarrier state.
 * @param phase_bit Phase parity bit to test.
 * @return 1 if wait complete, 0 if not complete.
 */
/**
 * Attempt a parity-based try-wait on a shared mbarrier.
 * Returns 1 if the barrier is complete for the requested phase, 0 otherwise.
 * (Duplicate definition — behavior identical to the other overload.)
 * @param smem_barrier Shared-memory storage for mbarrier state.
 * @param phase_bit Phase parity bit to test.
 * @return 1 if wait complete, 0 if not complete.
 */
/**
 * Block until the shared mbarrier reaches the given phase parity.
 * Uses repeated try-wait/retry loops if the first try does not complete.
 * @param smem_barrier Shared-memory storage for mbarrier state.
 * @param phase_bit Phase parity bit to wait for.
 */
/**
 * Poll the shared mbarrier for the given phase parity in a non-blocking loop.
 * Yields short nanosleep intervals between attempts to reduce contention on
 * pre-Hopper architectures.
 * @param smem_barrier Shared-memory storage for mbarrier state.
 * @param phase_bit Phase parity bit to wait for.
 */
/**
 * Signal arrival of the calling participant to the shared mbarrier.
 * @param smem_barrier Shared-memory storage for mbarrier state.
 */
/**
 * Conditionally signal arrival of a remapped CTA participant to a cluster
 * mbarrier when `pred` is non-zero.
 * @param smem_barrier Shared-memory storage for mbarrier state.
 * @param cta_id CTA identifier used for remapping.
 * @param pred If non-zero, the arrival is performed; otherwise no-op.
 */
/**
 * Set an expected transaction byte-size on the shared mbarrier. This informs
 * the barrier about the size of an upcoming transaction.
 * @param smem_barrier Shared-memory storage for mbarrier state.
 * @param transaction_bytes Expected transaction size in bytes.
 */
/**
 * Signal arrival to a shared mbarrier and specify the expected transaction
 * byte-size in the same operation.
 * @param smem_barrier Shared-memory storage for mbarrier state.
 * @param transaction_bytes Expected transaction size in bytes.
 */
/**
 * Signal arrival to an mbarrier using the cp.async arrival form (used with
 * cp.async-based transactions).
 * @param smem_barrier Shared-memory storage for mbarrier state.
 */
/**
 * Emit an async proxy fence to order shared/CTA asynchronous operations.
 */
/**
 * Indicate the issuing warp has finished enqueueing a TMA store group by
 * committing the cp.async bulk group.
 */
/**
 * Wait for a cp.async bulk CTA group completion of `Count` entries.
 * @tparam Count Number of entries to wait for (compile-time constant).
 */
/**
 * Perform a partial CTA-level synchronization using a shared mbarrier:
 * arrive then spin until the barrier advances.
 * @param smem_barrier Shared-memory storage for mbarrier state.
 */
/**
 * Increase the warp/group register allocation limit by `RegCount` registers.
 * @tparam RegCount Number of registers to allocate (compile-time constant).
 */
/**
 * Decrease the warp/group register allocation limit by `RegCount` registers.
 * @tparam RegCount Number of registers to deallocate (compile-time constant).
 */
namespace tl {
enum class CacheHintSm90 : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};

TL_DEVICE void tma_load(void *smem_ptr, void *gmem_ptr, uint64_t &smem_mbar,
                        uint32_t size) {
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3]; \n" ::"r"(smem_int_ptr),
               "l"(gmem_ptr), "r"(size), "r"(smem_int_mbar)
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

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, uint64_t &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes"
                 " [%0], [%1, {%3}], [%2];"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                   "r"(crd0)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1, {%3}], [%2], %4;"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                   "r"(crd0), "l"(cache_hint)
                 : "memory");
  }
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, uint64_t &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes"
                 " [%0], [%1, {%3, %4}], [%2];"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                   "r"(crd0), "r"(crd1)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1, {%3, %4}], [%2], %5;"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                   "r"(crd0), "r"(crd1), "l"(cache_hint)
                 : "memory");
  }
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, uint64_t &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes"
                 " [%0], [%1, {%3, %4, %5}], [%2];"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                   "r"(crd0), "r"(crd1), "r"(crd2)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1, {%3, %4, %5}], [%2], %6;"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                   "r"(crd0), "r"(crd1), "r"(crd2), "l"(cache_hint)
                 : "memory");
  }
}
template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, uint64_t &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2,
                        int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes"
                 " [%0], [%1, {%3, %4, %5, %6}], [%2];"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                   "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1, {%3, %4, %5, %6}], [%2], %7;"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                   "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "l"(cache_hint)
                 : "memory");
  }
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_load(const CUtensorMap &descriptor, uint64_t &smem_mbar,
                        void const *const smem_ptr, int32_t const &crd0,
                        int32_t const &crd1, int32_t const &crd2,
                        int32_t const &crd3, int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes"
                 " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                   "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.mbarrier::"
                 "complete_tx::bytes.L2::cache_hint"
                 " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;"
                 :
                 : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
                   "r"(crd0), "r"(crd1), "r"(crd2), "r"(crd3), "r"(crd4),
                   "l"(cache_hint)
                 : "memory");
  }
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_load_im2col(const CUtensorMap &descriptor,
                               uint64_t &smem_mbar, void const *const smem_ptr,
                               int32_t const &coord_c, int32_t const &coord_w,
                               int32_t const &coord_h, int32_t const &coord_n,
                               uint16_t const &offset_w,
                               uint16_t const &offset_h) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_mbar = smem_ptr_to_uint(&smem_mbar);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier:"
        ":complete_tx::bytes"
        " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8};"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
          "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n), "h"(offset_w),
          "h"(offset_h)
        : "memory");
  } else {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier:"
        ":complete_tx::bytes.L2::cache_hint"
        " [%0], [%1, {%3, %4, %5, %6}], [%2], {%7, %8}, %9;"
        :
        : "r"(smem_int_ptr), "l"(gmem_int_desc), "r"(smem_int_mbar),
          "r"(coord_c), "r"(coord_w), "r"(coord_h), "r"(coord_n), "h"(offset_w),
          "h"(offset_h), "l"(cache_hint)
        : "memory");
  }
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);

  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, "
                 "{%2}], [%1];"
                 :
                 : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group "
                 "::cache_hint [%0, {%2}], [%1], %3;"
                 :
                 : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0),
                   "l"(cache_hint)
                 : "memory");
  }
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);

  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, "
                 "{%2, %3}], [%1];"
                 :
                 : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group "
                 "::cache_hint [%0, {%2, %3}], [%1], %4;"
                 :
                 : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                   "l"(cache_hint)
                 : "memory");
  }
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1, int32_t const &crd2) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);

  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group [%0, "
                 "{%2, %3, %4}], [%1];"
                 :
                 : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                   "r"(crd2)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.bulk_group "
                 "::cache_hint [%0, {%2, %3, %4}], [%1], %5;"
                 :
                 : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                   "r"(crd2), "l"(cache_hint)
                 : "memory");
  }
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1, int32_t const &crd2,
                         int32_t const &crd3) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);

  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group [%0, "
                 "{%2, %3, %4, %5}], [%1];"
                 :
                 : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                   "r"(crd2), "r"(crd3)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.bulk_group "
                 "::cache_hint [%0, {%2, %3, %4, %5}], [%1], %6;"
                 :
                 : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                   "r"(crd2), "r"(crd3), "l"(cache_hint)
                 : "memory");
  }
}

template <CacheHintSm90 cache_hint = CacheHintSm90::EVICT_NORMAL>
TL_DEVICE void tma_store(const CUtensorMap &descriptor,
                         void const *const smem_ptr, int32_t const &crd0,
                         int32_t const &crd1, int32_t const &crd2,
                         int32_t const &crd3, int32_t const &crd4) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);

  if constexpr (cache_hint == CacheHintSm90::EVICT_NORMAL) {
    asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, "
                 "{%2, %3, %4, %5, %6}], [%1];"
                 :
                 : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                   "r"(crd2), "r"(crd3), "r"(crd4)
                 : "memory");
  } else {
    asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group "
                 "::cache_hint [%0, {%2, %3, %4, %5, %6}], [%1], %7;"
                 :
                 : "l"(gmem_int_desc), "r"(smem_int_ptr), "r"(crd0), "r"(crd1),
                   "r"(crd2), "r"(crd3), "r"(crd4), "l"(cache_hint)
                 : "memory");
  }
}

TL_DEVICE void prefetch_tma_descriptor(const CUtensorMap &descriptor) {
  uint64_t gmem_int_desc = reinterpret_cast<uint64_t>(&descriptor);
  asm volatile("prefetch.tensormap [%0];" : : "l"(gmem_int_desc) : "memory");
}

TL_DEVICE void mbarrier_init(uint64_t &smem_barrier, uint32_t arrive_count) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.init.shared.b64 [%1], %0;"
               :
               : "r"(arrive_count), "r"(smem_int_ptr));
}

TL_DEVICE uint32_t mbarrier_try_wait(uint64_t &smem_barrier, int phase_bit) {

  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  uint32_t waitComplete;

  asm volatile("{\n\t"
               ".reg .pred P1; \n\t"
               "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2; \n\t"
               "selp.b32 %0, 1, 0, P1; \n\t"
               "}"
               : "=r"(waitComplete)
               : "r"(smem_int_ptr), "r"(phase_bit));

  return waitComplete;
}

TL_DEVICE uint32_t mbarrier_try_wait(uint64_t &smem_barrier, int phase_bit) {

  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  uint32_t waitComplete;

  asm volatile("{\n\t"
               ".reg .pred P1; \n\t"
               "mbarrier.try_wait.parity.shared.b64 P1, [%1], %2; \n\t"
               "selp.b32 %0, 1, 0, P1; \n\t"
               "}"
               : "=r"(waitComplete)
               : "r"(smem_int_ptr), "r"(phase_bit));

  return waitComplete;
}

TL_DEVICE void mbarrier_wait(uint64_t &smem_barrier, int phase_bit) {
  if (mbarrier_try_wait(smem_barrier, phase_bit) == 0) {
    uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
    // Arbitrarily large timer value after which try-wait expires and re-tries.
    uint32_t ticks = 0x989680;
    asm volatile("{\n\t"
                 ".reg .pred       P1; \n\t"
                 "LAB_WAIT: \n\t"
                 "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1, %2; \n\t"
                 "@P1 bra DONE; \n\t"
                 "bra     LAB_WAIT; \n\t"
                 "DONE: \n\t"
                 "}"
                 :
                 : "r"(smem_int_ptr), "r"(phase_bit), "r"(ticks));
  }
}

TL_DEVICE void mbarrier_test_wait(uint64_t &smem_barrier, int phase_bit) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile(
      "{\n"
      ".reg .pred                P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.test_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1                       bra.uni DONE;\n"
      "nanosleep.u32 5;\n" // wait a few nanoseconds on pre-Hopper architectures
                           // to save instruction issue slots
      "bra.uni                   LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(smem_int_ptr),
      "r"(phase_bit));
}

TL_DEVICE void mbarrier_arrive(uint64_t &smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" : : "r"(smem_int_ptr));
}

TL_DEVICE void mbarrier_arrive(uint64_t &smem_barrier, int cta_id,
                               uint32_t pred) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  if (pred) {
    asm volatile("{\n\t"
                 ".reg .b32 remAddr32;\n\t"
                 "mapa.shared::cluster.u32  remAddr32, %0, %1;\n\t"
                 "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t"
                 "}"
                 :
                 : "r"(smem_int_ptr), "r"(cta_id));
  }
}

TL_DEVICE void mbarrier_expect_tx(uint64_t &smem_barrier,
                                  uint32_t transaction_bytes) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.expect_tx.shared.b64 [%1], %0;"
               :
               : "r"(transaction_bytes), "r"(smem_int_ptr));
}

TL_DEVICE void mbarrier_arrive_expect_tx(uint64_t &smem_barrier,
                                         uint32_t transaction_bytes) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%1], %0;"
               :
               : "r"(transaction_bytes), "r"(smem_int_ptr));
}

TL_DEVICE void mbarrier_cp_async_arrive(uint64_t &smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];"
               :
               : "r"(smem_int_ptr));
}

TL_DEVICE void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" : :);
}

// Indicate arrival of warp issuing TMA_STORE
TL_DEVICE void tma_store_arrive() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int Count> TL_DEVICE void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(Count) : "memory");
}

TL_DEVICE void syncthreads_partial(uint64_t &smem_barrier) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(&smem_barrier);
  uint64_t state = 0;
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "mbarrier.arrive.shared.b64 %1, [%0];\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.shared.b64 P1, [%0], %1;\n"
               "@!P1                      bra.uni LAB_WAIT;\n"
               "}\n"
               :
               : "r"(smem_int_ptr), "l"(state));
}

template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

} // namespace tl