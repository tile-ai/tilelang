/*!
 * \file tl/backend/webgpu/op/copy.cc
 * \brief WebGPU implementation for tl.copy lowering.
 */

#include "op/copy.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace webgpu {

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

} // namespace webgpu

namespace {

bool MatchWebGPUCopyTarget(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "webgpu";
}

bool RegisterWebGPUCopy() {
  RegisterCopyImpl(CopyImpl{
      "webgpu.Copy",
      MatchWebGPUCopyTarget,
      100,
      webgpu::Copy::InferLayout,
      webgpu::Copy::Lower,
  });
  return true;
}

const bool webgpu_copy_registered = RegisterWebGPUCopy();

} // namespace

} // namespace tl
} // namespace tvm
