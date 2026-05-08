/*!
 * \file tl/backend/rocm/op/fill.cc
 * \brief ROCm implementation for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchROCmFillTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmFill() {
  RegisterFillImpl(FillImpl{
      "rocm.Fill",
      MatchROCmFillTarget,
      backend::Fill::Lower,
  });
  return true;
}

const bool rocm_fill_registered = RegisterROCmFill();

} // namespace

} // namespace tl
} // namespace tvm
