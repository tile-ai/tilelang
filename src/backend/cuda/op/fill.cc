/*!
 * \file tl/backend/cuda/op/fill.cc
 * \brief CUDA implementation for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchCudaFillTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaFill() {
  RegisterFillImpl(FillImpl{
      "cuda.Fill",
      MatchCudaFillTarget,
      backend::Fill::Lower,
  });
  return true;
}

const bool cuda_fill_registered = RegisterCudaFill();

} // namespace

} // namespace tl
} // namespace tvm
