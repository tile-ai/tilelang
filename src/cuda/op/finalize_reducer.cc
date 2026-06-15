/*!
 * \file tl/cuda/op/finalize_reducer.cc
 * \brief CUDA implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/finalize_reducer.h"

#include "cuda/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace cuda {

struct FinalizeReducer : backend::FinalizeReducerLowerer<FinalizeReducer> {
  static int WarpSize(Target target) { return TargetCudaGetWarpSize(target); }

  // See reduce.cc::Reduce::BarrierIdFor for rationale (paired-ID scheme).
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

bool MatchCudaFinalizeReducerTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "cuda.FinalizeReducer",
      MatchCudaFinalizeReducerTarget,
      cuda::FinalizeReducer::Lower,
  });
  return true;
}

const bool cuda_finalize_reducer_registered = RegisterCudaFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
