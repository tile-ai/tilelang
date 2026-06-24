#pragma once

#include "common.h"
#include "cutlass/cutlass.h"

#if __CUDA_ARCH_LIST__ >= 900
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/mma_sm90_gmma.hpp"
#endif

namespace tl {

namespace detail {

// Provide architecture-specific defaults so callers may omit arguments.
TL_DEVICE constexpr int default_warp_size() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP_DEVICE_COMPILE__)
  return 64;
#else
  return 32;
#endif
}

TL_DEVICE constexpr int default_warps_per_group() { return 4; }

TL_DEVICE int linear_thread_idx_in_block() {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
#else
  return 0;
#endif
}

} // namespace detail

TL_DEVICE int get_lane_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() % warp_size;
}

TL_DEVICE int get_warp_idx_sync(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int get_warp_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int
get_warp_group_idx(int warp_size = detail::default_warp_size(),
                   int warps_per_group = detail::default_warps_per_group()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  warps_per_group =
      warps_per_group > 0 ? warps_per_group : detail::default_warps_per_group();
  int threads_per_group = warp_size * warps_per_group;
  threads_per_group = threads_per_group > 0 ? threads_per_group : warp_size;
  return detail::linear_thread_idx_in_block() / threads_per_group;
}

#if __CUDA_ARCH_LIST__ >= 900
TL_DEVICE void warpgroup_arrive() { cute::warpgroup_arrive(); }
TL_DEVICE void warpgroup_commit_batch() { cute::warpgroup_commit_batch(); }

template <int NumMma> TL_DEVICE void warpgroup_wait() {
  cute::warpgroup_wait<NumMma>();
}

TL_DEVICE void warpgroup_fence_operand(uint32_t *regs, int count) {
#pragma unroll
  for (int i = 0; i < count; ++i) {
    cute::warpgroup_fence_operand(regs[i]);
  }
}

TL_DEVICE void warpgroup_fence_operand(float *regs, int count) {
#pragma unroll
  for (int i = 0; i < count; ++i) {
    cute::warpgroup_fence_operand(regs[i]);
  }
}

// Template parameter:
//   thread_extent: the logical size (in number of threads) of each "group"
//                  within which we want to elect exactly ONE representative
//                  thread.
template <int thread_extent> TL_DEVICE bool tl_shuffle_elect() {
  // Special case: thread_extent == 0 means "elect exactly one thread
  // in the entire thread block", i.e., the leader of the first warp of the
  // block.
  if constexpr (thread_extent == 0) {
    // cutlass::canonical_warp_idx():
    //   Returns the warp ID within the thread block in a "canonical" way
    //   (0 for the first warp, 1 for the second, ...).
    // cute::elect_one_sync():
    //   Elect exactly one lane in the warp to return true (typically lane 0),
    //   other lanes return false.
    // The condition ensures that:
    //   (1) We are in warp 0 of the block.
    //   (2) We are the elected lane in this warp.
    // NOTE: we prefer canonical_warp_idx to canonical_warp_idx_sync here,
    //       because the latter uses shfl.sync to broadcast the value from lane
    //       0 to the whole warp to ensure that ptxas knows it is warp-uniform,
    //       which helps the register allocator to assign the value to a uniform
    //       register. However, shfl.sync is dispatched to the LSU pipe, causing
    //       contention with other SMEM traffic, which can cause serious
    //       performance degradation in SMEM-intensive kernels.
    //       Here, we use elect.sync to inform ptxas that we are executing this
    //       with only one thread, so we do not need shfl.sync to make it
    //       warp-uniform.
    return cute::elect_one_sync() && cutlass::canonical_warp_idx() == 0;
  } else if constexpr (thread_extent == 32) {
    return cute::elect_one_sync();
  } else {
    // General case: thread_extent != 0
    // We select warps with multiple of ceil(thread_extent / 32) warp IDs.
    // NOTE: we use canonical_warp_idx for the same reason as above.
    constexpr int warp_extent = (thread_extent + 31) / 32;
    static_assert(warp_extent > 0);
    return cute::elect_one_sync() &&
           (cutlass::canonical_warp_idx() % warp_extent) == 0;
  }
}

template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}
#endif

} // namespace tl
