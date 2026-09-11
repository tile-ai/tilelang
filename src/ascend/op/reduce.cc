/*!
 * \file tl/ascend/op/reduce.cc
 * \brief Ascend implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "backend/common/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace ascend {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool AllReduceNeedsWorkspace(int reducing_threads, int, Target) {
    // CheckAllReduceWidth ensures scale divides reducing_threads, so a
    // power-of-two width also has a power-of-two scale.
    return reducing_threads > 32 ||
           (reducing_threads & (reducing_threads - 1)) != 0;
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
