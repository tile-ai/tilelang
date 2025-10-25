#pragma once

#include "common.h"
#include <cutlass/arch/barrier.h>

// Reuse cutlass advanced barrier abstraction
using Barrier = cutlass::arch::ClusterTransactionBarrier;

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

template <typename BarrierType = uint64_t>
TL_DEVICE void mbarrier_cp_async_arrive_noinc(BarrierType &smem_mbar) {
  uint32_t smem_int_mbar;
  if constexpr (std::is_pointer_v<BarrierType>) {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(smem_mbar));
  } else {
    smem_int_mbar = smem_ptr_to_uint(reinterpret_cast<uint64_t *>(&smem_mbar));
  }
  asm volatile("{\n\t"
               "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
               "}"
               :
               : "r"(smem_int_mbar));
  cutlass::arch::synclog_emit_cpasync_barrier_arrive(__LINE__, smem_int_mbar);
}

TL_DEVICE void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" : :);
}

TL_DEVICE void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;" : :);
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
