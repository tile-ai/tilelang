/*!
 * \file tl/webgpu/op/copy.cc
 * \brief WebGPU implementation for tl.copy lowering.
 */

#include "op/copy.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace webgpu {

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    return op.InferSIMTLayout(layout_args, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    return LowerNormalCopy(op, lower_args, analyzer);
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
