/*!
 * \file tl/cpu/op/transpose.cc
 * \brief CPU implementation for tl.transpose lowering.
 */

#include "op/transpose.h"
#include <tvm/runtime/logging.h>

#include "backend/common/target_utils.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

namespace cpu {

struct Transpose {
  static Stmt Lower(const TransposeNode &op, const LowerArgs &lower_args,
                    sym::Analyzer analyzer) {
    if (!(IsLocalBuffer(op.src, true) || IsGlobalBuffer(op.src)) ||
        !(IsLocalBuffer(op.dst, true) || IsGlobalBuffer(op.dst))) {
      LOG(FATAL) << "CPU transpose only supports local and global buffers, but "
                 << "got src scope `" << op.src.scope().name()
                 << "` and dst scope `" << op.dst.scope().name() << "`.";
    }
    auto simt_loop = op.MakeSIMTLoop(analyzer);
    auto fused_loop = ParallelLoopFuser::Fuse(simt_loop).as_or_throw<For>();
    if (lower_args.target->kind->name == "c") {
      return fused_loop;
    }
    return VectorizeLoop(fused_loop, analyzer, lower_args.layout_map);
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
