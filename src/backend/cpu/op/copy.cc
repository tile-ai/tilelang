/*!
 * \file tl/backend/cpu/op/copy.cc
 * \brief CPU implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace cpu {

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    CopyInst copy_inst =
        SelectInst(op, T.target, T.layout_map, T.analyzer, T.buffer_oob);
    return CopyLoweringAccess::InferLayoutForCopyInst(op, T, level, copy_inst);
  }

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer, bool buffer_oob) {
    if (op.GetIsTmaCopy()) {
      LOG(FATAL) << "T.tma_copy() is not supported on CPU target "
                 << target->ToDebugString();
    }
    if (op.GetIsAsyncCopy() || op.GetNoImplicitAsyncCommitWait()) {
      LOG(FATAL) << "Async copy is not supported on CPU target "
                 << target->ToDebugString();
    }
    return CopyInst::kNormal;
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    auto copy_inst =
        SelectInst(op, T.target, T.layout_map, analyzer, /*buffer_oob=*/false);
    ICHECK(copy_inst == CopyInst::kNormal)
        << "Unsupported CPU copy inst " << static_cast<int>(copy_inst);
    return LowerNormalCopy(op, T, analyzer);
  }
};

} // namespace cpu

namespace {

bool MatchCPUCopyTarget(Target target) { return TargetIsCPU(target); }

bool RegisterCPUCopy() {
  RegisterCopyImpl(CopyImpl{
      "cpu.Copy",
      MatchCPUCopyTarget,
      100,
      cpu::Copy::InferLayout,
      cpu::Copy::SelectInst,
      cpu::Copy::Lower,
  });
  return true;
}

const bool cpu_copy_registered = RegisterCPUCopy();

} // namespace

} // namespace tl
} // namespace tvm
