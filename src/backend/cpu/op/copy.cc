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

namespace {

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetIsTmaCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "is_tma_copy");
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  return GetBoolAnnotation(op, "force_cp_async");
}

bool GetNoImplicitAsyncCommitWait(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kAsyncCopyNoImplicitCommitWait);
}

} // namespace

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    CheckSupported(op, T.target);
    return op.InferSIMTLayout(T, level);
  }

  static void CheckSupported(const CopyNode &op, Target target) {
    if (GetIsTmaCopy(op)) {
      LOG(FATAL) << "T.tma_copy() is not supported on CPU target "
                 << target->ToDebugString();
    }
    if (GetIsAsyncCopy(op) || GetNoImplicitAsyncCommitWait(op)) {
      LOG(FATAL) << "Async copy is not supported on CPU target "
                 << target->ToDebugString();
    }
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    CheckSupported(op, T.target);
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
