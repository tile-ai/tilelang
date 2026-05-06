/*!
 * \file tl/backend/rocm/op/gemm_sp.cc
 * \brief ROCm implementation boundary for tl.gemm_sp.
 */

#include "op/gemm_sp.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace rocm {

struct GemmSP {
  static std::pair<int, int> ComputeWarpPartition(const GemmSPWarpPolicyNode &,
                                                  int, int, int, Target target,
                                                  String, int) {
    LOG(FATAL) << "GemmSP is not supported on ROCm target: " << target->str();
    return {0, 0};
  }

  static Stmt Lower(const GemmSPNode &, const LowerArgs &T, arith::Analyzer *) {
    LOG(FATAL) << "GemmSP is not supported on ROCm target: " << T.target->str();
    return Stmt();
  }

  static LayoutMap InferLayout(const GemmSPNode &, const LayoutInferArgs &T,
                               InferLevel) {
    LOG(FATAL) << "GemmSP is not supported on ROCm target: " << T.target->str();
    return LayoutMap();
  }
};

} // namespace rocm

namespace {

bool MatchROCmGemmSPTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmGemmSP() {
  RegisterGemmSPImpl(GemmSPImpl{
      "rocm.GemmSP",
      MatchROCmGemmSPTarget,
      rocm::GemmSP::ComputeWarpPartition,
      rocm::GemmSP::Lower,
      rocm::GemmSP::InferLayout,
  });
  return true;
}

const bool rocm_gemm_sp_registered = RegisterROCmGemmSP();

} // namespace

} // namespace tl
} // namespace tvm
