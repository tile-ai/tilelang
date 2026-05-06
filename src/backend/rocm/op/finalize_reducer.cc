/*!
 * \file tl/backend/rocm/op/finalize_reducer.cc
 * \brief ROCm implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "op/finalize_reducer.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace rocm {

struct FinalizeReducer {
  static int WarpSize(Target) { return 64; }

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
      rocm::FinalizeReducer::WarpSize,
      rocm::FinalizeReducer::MakeBatchAllReduce,
      rocm::FinalizeReducer::MakeScalarAllReduce,
  });
  return true;
}

const bool rocm_finalize_reducer_registered = RegisterROCmFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
