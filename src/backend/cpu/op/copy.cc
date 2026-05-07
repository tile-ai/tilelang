/*!
 * \file tl/backend/cpu/op/copy.cc
 * \brief CPU implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "op/utils.h"
#include "target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace cpu {

namespace {

bool IsSupportedCPUBuffer(const Buffer &buffer) {
  return IsLocalBuffer(buffer, true) || IsGlobalBuffer(buffer);
}

void CheckSupportedCPUScope(const CopyNode &op) {
  if (!IsSupportedCPUBuffer(op.src) || !IsSupportedCPUBuffer(op.dst)) {
    LOG(FATAL) << "CPU copy only supports local and global buffers, but got "
               << "src scope `" << op.src.scope() << "` and dst scope `"
               << op.dst.scope() << "`.";
  }
}

} // namespace

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    (void)T;
    (void)level;
    CheckSupportedCPUScope(op);
    return {};
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    CheckSupportedCPUScope(op);
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
      cpu::Copy::Lower,
  });
  return true;
}

const bool cpu_copy_registered = RegisterCPUCopy();

} // namespace

} // namespace tl
} // namespace tvm
