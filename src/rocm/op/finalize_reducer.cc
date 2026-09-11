/*!
 * \file tl/rocm/op/finalize_reducer.cc
 * \brief ROCm implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/finalize_reducer.h"

#include "rocm/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace rocm {

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

  static int WarpSize(Target target) { return TargetRocmGetWarpSize(target); }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset, PrimExpr,
                                        int batch, int workspace_stride,
                                        Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ", " << batch << ", "
       << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset, PrimExpr,
                                         Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ">::run";
    return ss.str();
  }
};

} // namespace rocm

namespace {

bool MatchROCmFinalizeReducerTarget(Target target) {
  return TargetIsRocm(target);
}

bool RegisterROCmFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "rocm.FinalizeReducer",
      MatchROCmFinalizeReducerTarget,
      rocm::FinalizeReducer::Lower,
  });
  return true;
}

const bool rocm_finalize_reducer_registered = RegisterROCmFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
