/*!
 * \file tl/ascend/op/gemm_blockscaled.cc
 * \brief Ascend instruction selection for block-scaled GEMM.
 */

#include "op/gemm_blockscaled.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

using namespace ffi;

namespace ascend {
namespace {

// Instruction key resolved by the Python registry
// (tilelang/ascend/op/gemm/__init__.py) to GemmMADBlockScaled.
constexpr const char *kAscendMADBlockScaled = "ascend.mad.blockscaled";

String SelectBlockScaledGemmInst(const GemmBlockScaled &op, int block_size,
                                 const Target &target) {
  (void)op;
  (void)block_size;
  (void)target;
  return kAscendMADBlockScaled;
}

} // namespace
} // namespace ascend

namespace {

bool MatchAscendGemmBlockScaledTarget(Target target) {
  return TargetIsAscend(target);
}

bool RegisterAscendGemmBlockScaled() {
  RegisterGemmBlockScaledImpl(GemmBlockScaledImpl{
      "ascend.GemmBlockScaled",
      MatchAscendGemmBlockScaledTarget,
      ascend::SelectBlockScaledGemmInst,
  });
  return true;
}

const bool ascend_gemm_blockscaled_registered = RegisterAscendGemmBlockScaled();

} // namespace

} // namespace tl
} // namespace tvm
