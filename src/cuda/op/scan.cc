/*!
 * \file tl/cuda/op/scan.cc
 * \brief CUDA implementation registration for tl scan lowering.
 */

#include "backend/common/op/scan.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchCudaScanTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaScan() {
  RegisterCumSumImpl(CumSumImpl{
      "cuda.CumSum",
      MatchCudaScanTarget,
      backend::scan::LowerCumSum,
  });
  RegisterCumMaxImpl(CumMaxImpl{
      "cuda.CumMax",
      MatchCudaScanTarget,
      backend::scan::LowerCumMax,
  });
  return true;
}

const bool cuda_scan_registered = RegisterCudaScan();

} // namespace

} // namespace tl
} // namespace tvm
