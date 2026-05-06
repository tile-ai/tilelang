/*!
 * \file tl/backend/cpu/op/reduce.cc
 * \brief CPU implementation for tl.reduce AllReduce lowering.
 */

#include "op/reduce.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace cpu {

struct Reduce {
  static bool SupportsFp16Bf16NanReduce(Target) { return false; }

  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    return op.LowerWithAllReduce(T, analyzer,
                                 SupportsFp16Bf16NanReduce(T.target),
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

} // namespace cpu

namespace {

bool MatchCPUReduceTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUReduce() {
  RegisterReduceImpl(ReduceImpl{
      "cpu.Reduce",
      MatchCPUReduceTarget,
      cpu::Reduce::Lower,
  });
  return true;
}

const bool cpu_reduce_registered = RegisterCPUReduce();

} // namespace

} // namespace tl
} // namespace tvm
