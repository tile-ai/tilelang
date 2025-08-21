#pragma once

#include "common.h"
#include <cutlass/arch/barrier.h>

// Reuse cutlass advanced barrier abstraction
using Barrier = cutlass::arch::ClusterTransactionBarrier;

/**
 * Initialize a shared-memory transaction barrier.
 *
 * Initializes the barrier object referenced by `smem_barrier` with the specified
 * number of arrivals required to complete the barrier.
 *
 * @param smem_barrier Reference to the barrier stored in shared memory.
 * @param arrive_count Number of arrivals required for the barrier to complete.
 */

/**
 * Try to complete a barrier wait without blocking.
 *
 * Performs a non-blocking probe of the barrier's parity for the given phase.
 *
 * @param smem_barrier Reference to the barrier stored in shared memory.
 * @param phase_bit Phase/parity bit to test (usually 0 or 1).
 * @return 1 if the wait is already satisfied (barrier completed for this phase), otherwise 0.
 */

/**
 * Blocking wait on a shared-memory transaction barrier.
 *
 * Blocks the calling context until the barrier referenced by `smem_barrier`
 * advances for the specified `phase_bit`. Internally uses a try-wait loop
 * and will spin until completion.
 *
 * @param smem_barrier Reference to the barrier stored in shared memory.
 * @param phase_bit Phase/parity bit to wait on (usually 0 or 1).
 */

/**
 * Test-wait on a barrier with cooperative yielding.
 *
 * Repeatedly tests the barrier parity for `phase_bit` and yields briefly
 * between attempts (uses nanosleep on pre-Hopper architectures) to reduce
 * busy-wait pressure.
 *
 * @param smem_barrier Reference to the barrier stored in shared memory.
 * @param phase_bit Phase/parity bit to test (usually 0 or 1).
 */

/**
 * Signal arrival to the local shared-memory barrier.
 *
 * Marks this context as having arrived at the barrier referenced by
 * `smem_barrier`.
 *
 * @param smem_barrier Reference to the barrier stored in shared memory.
 */

/**
 * Conditionally signal arrival to a cluster-local barrier for a given CTA.
 *
 * If `pred` is non-zero, computes the cluster-shared address for `smem_barrier`
 * for the CTA specified by `cta_id` and signals arrival on that remote barrier.
 *
 * @param smem_barrier Reference to the barrier stored in shared memory.
 * @param cta_id Cluster CTA identifier whose barrier instance should be targeted.
 * @param pred Predicate; arrival is performed only if non-zero.
 */

/**
 * Set the expected transaction size for the barrier.
 *
 * Informs the barrier of the number of transaction bytes that will be issued,
 * used to track in-flight transactional work associated with the barrier.
 *
 * @param smem_barrier Reference to the barrier stored in shared memory.
 * @param transaction_bytes Expected transaction size in bytes.
 */

/**
 * Signal arrival and indicate expected transaction size in a single operation.
 *
 * Combines an arrival with a provided expected transaction byte count.
 *
 * @param smem_barrier Reference to the barrier stored in shared memory.
 * @param transaction_bytes Expected transaction size in bytes.
 */

/**
 * Issue an asynchronous cp.async arrival to a shared-memory barrier.
 *
 * Enqueues a cp.async mbarrier arrival for the barrier referenced by
 * `smem_mbar`. `BarrierType` may be a pointer or a barrier object type.
 *
 * @param smem_mbar Barrier (or pointer to barrier) located in shared memory.
 */

/**
 * Issue a fence.proxy.async for shared-memory proxy operations.
 *
 * Ensures ordering for prior asynchronous proxy operations.
 */

/**
 * Indicate arrival of a warp involved in a TMA_STORE bulk CP async operation.
 *
 * Marks completion of a warp's contribution to the current cp.async bulk group.
 */

/**
 * Wait for a TMA_STORE cp.async bulk group to complete.
 *
 * Blocks until the cp.async bulk read group identified by `Count` has finished.
 *
 * @tparam Count The cp.async bulk read group identifier/count to wait for (compile-time constant).
 */

/**
 * Partial-threadgroup synchronization using a shared barrier.
 *
 * Performs a single arrival on `smem_barrier` and then spins with try-wait
 * until the barrier for the arriving subset completes. Intended for
 * synchronizing a subset of threads without full __syncthreads().
 *
 * @param smem_barrier Reference to the barrier stored in shared memory.
 */
namespace tl {

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

template <typename BarrierType = uint64_t>
TL_DEVICE void mbarrier_cp_async_arrive(BarrierType &smem_mbar) {
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  asm volatile("cp.async.mbarrier.arrive.shared.b64 [%0];"
               :
               : "r"(smem_int_mbar));
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
} // namespace tl
