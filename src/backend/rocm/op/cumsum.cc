/*!
 * \file tl/backend/rocm/op/cumsum.cc
 * \brief ROCm implementation for tl.cumsum lowering.
 */

#include "backend/common/op/cumsum.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchROCmCumSumTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmCumSum() {
  RegisterCumSumImpl(CumSumImpl{
      "rocm.CumSum",
      MatchROCmCumSumTarget,
      backend::CumSum::Lower,
  });
  return true;
}

const bool rocm_cumsum_registered = RegisterROCmCumSum();

} // namespace

} // namespace tl
} // namespace tvm
