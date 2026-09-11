/*!
 * \file tl/rocm/op/reduce.cc
 * \brief ROCm implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "backend/common/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace rocm {

struct Reduce : backend::ReduceLowerer<Reduce> {
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

  static bool SupportsFp16Bf16NanReduce(Target) { return false; }

  static int GetPreferredVectorizedSize(const ReduceOpNode &, Target) {
    return 1;
  }

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
      rocm::Reduce::Lower,
  });
  return true;
}

const bool rocm_reduce_registered = RegisterROCmReduce();

} // namespace

} // namespace tl
} // namespace tvm
