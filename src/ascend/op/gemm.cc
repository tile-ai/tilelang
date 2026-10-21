/*!
 * \file tl/ascend/op/gemm.cc
 * \brief Ascend implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

using namespace ffi;
using namespace tirx;

namespace ascend {
namespace {

constexpr const char *kAscendMAD = "ascend.mad";

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)op;
    (void)block_size;
    (void)target;
    return kAscendMAD;
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
    return true;
  }
};

} // namespace ascend

namespace {

bool MatchAscendGemmTarget(Target target) { return TargetIsAscend(target); }

bool RegisterAscendGemm() {
  RegisterGemmImpl(GemmImpl{
      "ascend.Gemm",
      MatchAscendGemmTarget,
      ascend::Gemm::SelectInst,
      ascend::Gemm::ComputeWarpPartition,
      ascend::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool ascend_gemm_registered = RegisterAscendGemm();

} // namespace

} // namespace tl
} // namespace tvm
