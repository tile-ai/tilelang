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
    CheckSupported(op, T.target);
    return op.InferSIMTLayout(T, level);
  }

  static void CheckSupported(const CopyNode &op, Target target) {
    if (op.GetIsTmaCopy()) {
      LOG(FATAL) << "T.tma_copy() is not supported on CPU target "
                 << target->ToDebugString();
    }
    if (op.GetIsAsyncCopy() || op.GetNoImplicitAsyncCommitWait()) {
      LOG(FATAL) << "Async copy is not supported on CPU target "
                 << target->ToDebugString();
    }
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    CheckSupported(op, T.target);
    return LowerNormalCopy(op, T, analyzer);
  }

  static CopyInstructionKind ClassifyInstruction(const CopyNode &op,
                                                 Target target,
                                                 bool in_pipeline,
                                                 arith::Analyzer *analyzer) {
    if (op.GetIsAsyncCopy()) {
      return CopyInstructionKind::kCPAsync;
    }
    return CopyInstructionKind::kSync;
  }

  static CopyPipelineRole ClassifyPipelineRole(const CopyNode &op,
                                               Target target,
                                               arith::Analyzer *analyzer) {
    if (op.GetIsAsyncCopy()) {
      return CopyPipelineRole::kCPAsyncProducer;
    }
    return CopyPipelineRole::kConsumer;
  }

  static bool CanPipelineManageAsync(const CopyNode &op, Target target,
                                     arith::Analyzer *analyzer) {
    return false;
  }

  static bool IsSyncGlobalToSharedPrefix(const CopyNode &op, Target target,
                                         arith::Analyzer *analyzer) {
    return false;
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
      cpu::Copy::Lower,
      cpu::Copy::ClassifyInstruction,
      cpu::Copy::ClassifyPipelineRole,
      cpu::Copy::CanPipelineManageAsync,
      cpu::Copy::IsSyncGlobalToSharedPrefix,
  });
  return true;
}

const bool cpu_copy_registered = RegisterCPUCopy();

} // namespace

} // namespace tl
} // namespace tvm
