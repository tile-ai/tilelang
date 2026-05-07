/*!
 * \file tl/backend/rocm/op/transpose.cc
 * \brief ROCm implementation for tl.transpose lowering.
 */

#include "backend/common/op/transpose.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchROCmTransposeTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmTranspose() {
  RegisterTransposeImpl(TransposeImpl{
      "rocm.Transpose",
      MatchROCmTransposeTarget,
      backend::Transpose::Lower,
  });
  return true;
}

const bool rocm_transpose_registered = RegisterROCmTranspose();

} // namespace

} // namespace tl
} // namespace tvm
