/*!
 * \file tl/backend/webgpu/op/transpose.cc
 * \brief WebGPU implementation for tl.transpose lowering.
 */

#include "op/transpose.h"

#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

#include <dlpack/dlpack.h>
#include <vector>

namespace tvm {
namespace tl {

namespace webgpu {

struct Transpose {
  static Stmt Lower(const TransposeNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    bool is_cpu_target = T.target->GetTargetDeviceType() == kDLCPU;
    auto simt_loop = op.MakeSIMTLoop(analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

    if (is_cpu_target || IsLocalBuffer(op.src) || IsLocalBuffer(op.dst)) {
      return VectorizeLoop(fused_loop, T.layout_map);
    }

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
    return LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var,
                             analyzer, T.layout_map,
                             par_op->GetPredicate(T.thread_var));
  }
};

} // namespace webgpu

namespace {

bool MatchWebGPUTransposeTarget(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "webgpu";
}

bool RegisterWebGPUTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "webgpu.Transpose",
      MatchWebGPUTransposeTarget,
      webgpu::Transpose::Lower,
  });
  return true;
}

const bool webgpu_transpose_registered = RegisterWebGPUTranspose();

} // namespace

} // namespace tl
} // namespace tvm
