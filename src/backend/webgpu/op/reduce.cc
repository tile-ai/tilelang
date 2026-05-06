/*!
 * \file tl/backend/webgpu/op/reduce.cc
 * \brief WebGPU implementation for tl.reduce AllReduce lowering.
 */

#include "op/reduce.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace webgpu {

struct Reduce {
  static bool SupportsFp16Bf16NanReduce(Target) { return false; }

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

bool MatchWebGPUReduceTarget(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "webgpu";
}

bool RegisterWebGPUReduce() {
  RegisterReduceImpl(ReduceImpl{
      "webgpu.Reduce",
      MatchWebGPUReduceTarget,
      webgpu::Reduce::SupportsFp16Bf16NanReduce,
      webgpu::Reduce::MakeBatchAllReduce,
      webgpu::Reduce::MakeScalarAllReduce,
  });
  return true;
}

const bool webgpu_reduce_registered = RegisterWebGPUReduce();

} // namespace

} // namespace tl
} // namespace tvm
