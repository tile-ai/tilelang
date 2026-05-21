/*!
 * \file tl/backend/metal/op/gemm.cc
 * \brief Metal implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"
#include "target/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

namespace {

constexpr const char *kMetalScalar = "metal.scalar";

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)op;
    (void)block_size;
    (void)target;
    return kMetalScalar;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    (void)M;
    (void)N;
    (void)block_size;
    (void)target;
    (void)gemm_inst;
    policy.m_warp = 1;
    policy.n_warp = 1;
    return {1, 1};
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }

  static String InstructionKind(String gemm_inst) {
    (void)gemm_inst;
    return "scalar";
  }
};

} // namespace metal

namespace {

bool MatchMetalGemmTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalGemm() {
  RegisterGemmImpl(GemmImpl{
      "metal.Gemm",
      MatchMetalGemmTarget,
      metal::Gemm::SelectInst,
      metal::Gemm::ComputeWarpPartition,
      metal::Gemm::ReuseExistingSharedLayout,
      metal::Gemm::InstructionKind,
  });
  return true;
}

const bool metal_gemm_registered = RegisterMetalGemm();

} // namespace

} // namespace tl
} // namespace tvm
