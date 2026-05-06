/*!
 * \file tl/backend/webgpu/op/finalize_reducer.cc
 * \brief WebGPU implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "op/finalize_reducer.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace webgpu {

struct FinalizeReducer {
  static int WarpSize(Target) { return 32; }

  static Stmt Lower(const FinalizeReducerOpNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    return op.LowerWithAllReduce(T, analyzer, WarpSize(T.target),
                                 MakeBatchAllReduce, MakeScalarAllReduce);
  }

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

} // namespace webgpu

namespace {

bool MatchWebGPUFinalizeReducerTarget(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "webgpu";
}

bool RegisterWebGPUFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "webgpu.FinalizeReducer",
      MatchWebGPUFinalizeReducerTarget,
      webgpu::FinalizeReducer::Lower,
  });
  return true;
}

const bool webgpu_finalize_reducer_registered = RegisterWebGPUFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
