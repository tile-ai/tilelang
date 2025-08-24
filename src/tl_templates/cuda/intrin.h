#pragma once

#if __CUDA_ARCH_LIST__ >= 900
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/cutlass.h"

namespace tl {
// Template parameter:
//   thread_extent: the logical size (in number of threads) of each "group"
//                  within which we want to elect exactly ONE representative
//                  thread.
template <int thread_extent> TL_DEVICE bool tl_shuffle_elect() {

  // Special case: thread_extent == 0 means "elect exactly one thread
  // in the entire thread block", i.e., the leader of the first warp of the
  // block.
  if constexpr (thread_extent == 0) {
    // cutlass::canonical_warp_idx_sync():
    //   Returns the warp ID within the thread block in a "canonical" way
    //   (0 for the first warp, 1 for the second, ...).
    // cute::elect_one_sync():
    //   Elect exactly one lane in the warp to return true (typically lane 0),
    //   other lanes return false.
    // The condition ensures that:
    //   (1) We are in warp 0 of the block.
    //   (2) We are the elected lane in this warp.
    return cutlass::canonical_warp_idx_sync() == 0 && cute::elect_one_sync();
  }

  // General case: thread_extent != 0
  // (threadIdx.x / 32) is the warp index in the block.
  // (thread_extent / 32) is the number of warps in one group of size
  // thread_extent. We take warp_id % num_warps_in_group to get the warp's index
  // within the group.
  // __shfl_sync(mask, value, srcLane): broadcast 'value' from srcLane to all
  // lanes in the warp. Here it broadcasts the group-local warp index from lane
  // 0. Comparing to 0 selects only the group's warp 0.
  return __shfl_sync(0xffffffff, // full warp mask
                     (threadIdx.x / 32) %
                         (thread_extent / 32), // warp index within group
                     0                         // take the value from lane 0
                     ) == 0 &&
         // Within that group leader warp, elect exactly one lane (typically
         // lane 0) to be the single representative for the group.
         cute::elect_one_sync();
}

template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount> TL_DEVICE void warpgroup_reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}
} // namespace tl
#endif