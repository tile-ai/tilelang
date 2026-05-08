/*!
 * \file tl/backend/cuda/op/transpose.cc
 * \brief CUDA implementation for tl.transpose lowering.
 */

#include "backend/common/op/transpose.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchCudaTransposeTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "cuda.Transpose",
      MatchCudaTransposeTarget,
      backend::Transpose::Lower,
  });
  return true;
}

const bool cuda_transpose_registered = RegisterCudaTranspose();

} // namespace

} // namespace tl
} // namespace tvm
