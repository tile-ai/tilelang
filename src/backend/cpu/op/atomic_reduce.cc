/*!
 * \file tl/backend/cpu/op/atomic_reduce.cc
 * \brief CPU implementation for tl.atomicmax/tl.atomicmin lowering.
 */

#include "op/atomic_reduce.h"

#include "target/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include <vector>

namespace tvm {
namespace tl {

namespace cpu {

struct AtomicReduce {
  static Stmt Lower(const AtomicOpBaseNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    auto simt_loop = op.MakeSIMTLoop(analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    auto par_op = ParallelOp(fused_loop);
    std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                      InferLevel::kFree};
    for (auto level : levels) {
      par_op->InferLayout({T.target,
                           T.thread_bounds,
                           T.layout_map,
                           analyzer,
                           false,
                           T.buffer_remap,
                           {}},
                          level);
    }
    auto loop_layout = par_op->GetLoopLayout();
    return LowerParallelLoop(fused_loop, loop_layout, T.thread_var, analyzer,
                             T.layout_map, par_op->GetPredicate(T.thread_var));
  }
};

} // namespace cpu

namespace {

bool MatchCPUAtomicReduceTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUAtomicReduce() {
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "cpu.AtomicReduce",
      MatchCPUAtomicReduceTarget,
      cpu::AtomicReduce::Lower,
  });
  return true;
}

const bool cpu_atomic_reduce_registered = RegisterCPUAtomicReduce();

} // namespace

} // namespace tl
} // namespace tvm
