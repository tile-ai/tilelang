/*!
 * \file tl/backend/cpu/op/finalize_reducer.cc
 * \brief CPU implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "op/finalize_reducer.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace cpu {

struct FinalizeReducer {
  static int WarpSize(Target) { return 32; }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset, PrimExpr,
                                        int batch, int workspace_stride,
                                        Target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ", tl::SyncThreadsBarrier, "
       << batch << ", " << workspace_stride << ">::run_batch";
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

} // namespace cpu

namespace {

bool MatchCPUFinalizeReducerTarget(Target target) {
  return TargetIsCPU(target);
}

bool RegisterCPUFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "cpu.FinalizeReducer",
      MatchCPUFinalizeReducerTarget,
      cpu::FinalizeReducer::WarpSize,
      cpu::FinalizeReducer::MakeBatchAllReduce,
      cpu::FinalizeReducer::MakeScalarAllReduce,
  });
  return true;
}

const bool cpu_finalize_reducer_registered = RegisterCPUFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
