/*!
 * \file tl/rocm/op/scan.cc
 * \brief ROCm implementation registration for tl scan lowering.
 */

#include "backend/common/op/scan.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchROCmScanTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmScan() {
  RegisterCumSumImpl(CumSumImpl{
      "rocm.CumSum",
      MatchROCmScanTarget,
      backend::scan::LowerCumSum,
  });
  RegisterCumMaxImpl(CumMaxImpl{
      "rocm.CumMax",
      MatchROCmScanTarget,
      backend::scan::LowerCumMax,
  });
  return true;
}

const bool rocm_scan_registered = RegisterROCmScan();

} // namespace

} // namespace tl
} // namespace tvm
