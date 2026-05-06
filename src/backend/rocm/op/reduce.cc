/*!
 * \file tl/backend/rocm/op/reduce.cc
 * \brief ROCm implementation for tl.reduce AllReduce lowering.
 */

#include "op/reduce.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace rocm {

struct Reduce {
  static bool SupportsFp16Bf16NanReduce(Target) { return false; }

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

bool MatchROCmReduceTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmReduce() {
  RegisterReduceImpl(ReduceImpl{
      "rocm.Reduce",
      MatchROCmReduceTarget,
      rocm::Reduce::SupportsFp16Bf16NanReduce,
      rocm::Reduce::MakeBatchAllReduce,
      rocm::Reduce::MakeScalarAllReduce,
  });
  return true;
}

const bool rocm_reduce_registered = RegisterROCmReduce();

} // namespace

} // namespace tl
} // namespace tvm
