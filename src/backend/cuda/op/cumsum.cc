/*!
 * \file tl/backend/cuda/op/cumsum.cc
 * \brief CUDA implementation for tl.cumsum lowering.
 */

#include "backend/common/op/cumsum.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchCudaCumSumTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaCumSum() {
  RegisterCumSumImpl(CumSumImpl{
      "cuda.CumSum",
      MatchCudaCumSumTarget,
      backend::CumSum::Lower,
  });
  return true;
}

const bool cuda_cumsum_registered = RegisterCudaCumSum();

} // namespace

} // namespace tl
} // namespace tvm
