/*!
 * \file tl/webgpu/op/fill.cc
 * \brief WebGPU implementation for tl.fill lowering.
 */

#include "backend/common/op/fill.h"
#include <tvm/runtime/logging.h>

#include "op/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

namespace webgpu {

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &lower_args,
                    sym::Analyzer analyzer) {
    return backend::Fill::Lower(op, lower_args, analyzer);
  }
};

} // namespace webgpu

namespace {

bool MatchWebGPUFillTarget(Target target) {
  return target->kind->name == "webgpu";
}

bool RegisterWebGPUFill() {
  RegisterFillImpl(FillImpl{
      "webgpu.Fill",
      MatchWebGPUFillTarget,
      webgpu::Fill::Lower,
  });
  return true;
}

const bool webgpu_fill_registered = RegisterWebGPUFill();

} // namespace

} // namespace tl
} // namespace tvm
