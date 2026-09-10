/*!
 * \file tl/metal/op/reduce.cc
 * \brief Metal implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "metal/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static Array<PrimExpr> ThreadReduceArgs(const LowerArgs &args,
                                          const Fragment &layout, int threads) {
    if (threads > 32) {
      auto range = backend::reduce::ResolveAllReduceThreadRange(
          layout, args.thread_bounds, args.target);
      arith::Analyzer analyzer;
      ICHECK(analyzer.CanProveEqual(range->min, args.thread_bounds->min) &&
             analyzer.CanProveEqual(range->extent, args.thread_bounds->extent))
          << "Metal reductions across SIMD groups require all block threads "
             "to participate in the threadgroup barrier";
    }
    return {args.thread_index};
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

} // namespace metal

namespace {

bool MatchMetalReduceTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalReduce() {
  RegisterReduceImpl(ReduceImpl{
      "metal.Reduce",
      MatchMetalReduceTarget,
      metal::Reduce::Lower,
  });
  return true;
}

const bool metal_reduce_registered = RegisterMetalReduce();

} // namespace

} // namespace tl
} // namespace tvm
