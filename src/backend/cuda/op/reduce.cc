/*!
 * \file tl/backend/cuda/op/reduce.cc
 * \brief CUDA implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace cuda {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool SupportsFp16Bf16NanReduce(Target target) {
    return TargetIsCuda(target);
  }

  // Derive a unique pair of barrier IDs per concurrent WG so two WGs doing
  // inter-thread reductions don't collide on bar.sync. Each AllReduce uses
  // TWO barriers (sync<barrier_id> and sync<barrier_id+1>), so per-WG IDs
  // must be at least 2 apart. Map wg_idx -> 1 + 2*(wg_idx & 0x7): WG0 -> 1
  // (uses 1,2), WG1 -> 3 (uses 3,4), ..., WG7 -> 15 (uses 15, 0 = __syncthreads
  // which is fine because at most one WG can wrap onto 0). The mask keeps the
  // result in 1..15 (hardware has 16 named barriers).
  static int BarrierIdFor(PrimExpr thread_offset, int reducing_threads) {
    auto offset_int = as_const_int(thread_offset);
    if (offset_int == nullptr || reducing_threads <= 0) {
      return 1;
    }
    int wg_idx = static_cast<int>(*offset_int / reducing_threads);
    return 1 + 2 * (wg_idx & 0x7);
  }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetHasSMVersionGE(target, 90)) {
      ss << ", tl::NamedBarrier<" << all_threads << ", "
         << BarrierIdFor(thread_offset, reducing_threads) << ">";
    } else {
      ss << ", tl::SyncThreadsBarrier";
    }
    ss << ", " << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetHasSMVersionGE(target, 90)) {
      ss << ", tl::NamedBarrier<" << all_threads << ", "
         << BarrierIdFor(thread_offset, reducing_threads) << ">";
    }
    ss << ">::run";
    return ss.str();
  }
};

} // namespace cuda

namespace {

bool MatchCudaReduceTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaReduce() {
  RegisterReduceImpl(ReduceImpl{
      "cuda.Reduce",
      MatchCudaReduceTarget,
      cuda::Reduce::Lower,
  });
  return true;
}

const bool cuda_reduce_registered = RegisterCudaReduce();

} // namespace

} // namespace tl
} // namespace tvm
