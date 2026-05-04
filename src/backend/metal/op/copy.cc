/*!
 * \file tl/backend/metal/op/copy.cc
 * \brief Metal implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace metal {

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
      LOG(FATAL) << "T.tma_copy() is not supported on Metal target "
                 << target->ToDebugString();
    }
    if (GetIsAsyncCopy(op) || GetNoImplicitAsyncCommitWait(op)) {
      LOG(FATAL) << "Async copy is not supported on Metal target "
                 << target->ToDebugString();
    }
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    CheckSupported(op, T.target);
    return LowerNormalCopy(op, T, analyzer);
  }
};

} // namespace metal

namespace {

bool MatchMetalCopyTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalCopy() {
  RegisterCopyImpl(CopyImpl{
      "metal.Copy",
      MatchMetalCopyTarget,
      100,
      metal::Copy::InferLayout,
      metal::Copy::Lower,
  });
  return true;
}

const bool metal_copy_registered = RegisterMetalCopy();

} // namespace

} // namespace tl
} // namespace tvm
