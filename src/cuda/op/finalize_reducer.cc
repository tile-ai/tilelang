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
  static bool AllReduceNeedsWorkspace(int reducing_threads, int, Target) {
    return reducing_threads > 32;
  }

  // The XOR-butterfly shuffle all-reduce has no arbitrary-width fallback, so
  // the power-of-two rule applies on top of the universal checks.
  static void CheckAllReduceWidth(int reducing_threads, int scale,
                                  const char *op_name, Target) {
    backend::reduce::CheckAllReduceWidth(reducing_threads, scale, op_name);
    backend::reduce::CheckXorButterflyWidth(reducing_threads, scale);
  }

  static int WarpSize(Target target) { return TargetCudaGetWarpSize(target); }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetSupportsNamedBarrier(target)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
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
    if (TargetSupportsNamedBarrier(target)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
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
