/*!
 * \file tl/ascend/op/reduce.cc
 * \brief Ascend implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "ascend/op/ascend_allreduce_policy.h"
#include "backend/common/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace ascend {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool AllReduceNeedsWorkspace(int reducing_threads, int scale, Target) {
    return ascend::AllReduceNeedsWorkspace(reducing_threads, scale);
  }

  static bool SupportsFp16Bf16NanReduce(Target) { return false; }

  static int GetPreferredVectorizedSize(const ReduceOpNode &, Target) {
    return 1;
  }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset, PrimExpr, int,
                                        int, Target) {
    std::stringstream ss;
    ss << "tl::AscendAllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ">::run";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset, PrimExpr,
                                         Target) {
    std::stringstream ss;
    ss << "tl::AscendAllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset << ">::run";
    return ss.str();
  }
};

} // namespace ascend

namespace {

bool MatchAscendReduceTarget(Target target) { return TargetIsAscend(target); }

bool RegisterAscendReduce() {
  RegisterReduceImpl(ReduceImpl{
      "ascend.Reduce",
      MatchAscendReduceTarget,
      ascend::Reduce::Lower,
  });
  return true;
}

const bool ascend_reduce_registered = RegisterAscendReduce();

} // namespace

} // namespace tl
} // namespace tvm
