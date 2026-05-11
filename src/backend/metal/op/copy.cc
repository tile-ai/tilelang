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

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    return op.InferSIMTLayout(T, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
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
