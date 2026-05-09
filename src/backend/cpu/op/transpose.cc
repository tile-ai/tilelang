/*!
 * \file tl/backend/cpu/op/transpose.cc
 * \brief CPU implementation for tl.transpose lowering.
 */

#include "op/transpose.h"

#include "op/utils.h"
#include "target/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

namespace cpu {

struct Transpose {
  static Stmt Lower(const TransposeNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (!(IsLocalBuffer(op.src, true) || IsGlobalBuffer(op.src)) ||
        !(IsLocalBuffer(op.dst, true) || IsGlobalBuffer(op.dst))) {
      LOG(FATAL) << "CPU transpose only supports local and global buffers, but "
                 << "got src scope `" << op.src.scope() << "` and dst scope `"
                 << op.dst.scope() << "`.";
    }
    auto simt_loop = op.MakeSIMTLoop(analyzer);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
    return VectorizeLoop(fused_loop, T.layout_map);
  }
};

} // namespace cpu

namespace {

bool MatchCPUTransposeTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "cpu.Transpose",
      MatchCPUTransposeTarget,
      cpu::Transpose::Lower,
  });
  return true;
}

const bool cpu_transpose_registered = RegisterCPUTranspose();

} // namespace

} // namespace tl
} // namespace tvm
