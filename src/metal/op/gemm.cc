/*!
 * \file tl/metal/op/gemm.cc
 * \brief Metal implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "metal/op/utils.h"
#include "metal/target_utils.h"

#include <tvm/runtime/logging.h>

#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

namespace {

constexpr const char *kMetalSIMDGroup = "metal.simdgroup";

std::pair<int, int> ComputeMetalWarpPartition(const GemmWarpPolicyNode &policy,
                                              int M, int N, int num_warps) {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 8;
  constexpr int kNPerWarp = 8;

  TVM_FFI_ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  TVM_FFI_ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  if (policy.isFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * kMPerWarp) != 0) {
      int max_m_warps = M / kMPerWarp;
      m_warp = max_m_warps;
      n_warp = num_warps / m_warp;
      if (n_warp == 0) {
        n_warp = 1;
      }
    }
  } else if (policy.isFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * kNPerWarp) != 0) {
      int max_n_warps = N / kNPerWarp;
      n_warp = max_n_warps;
      m_warp = num_warps / n_warp;
      if (m_warp == 0) {
        m_warp = 1;
      }
    }
  } else if (policy.isSquare()) {
    std::tie(m_warp, n_warp) =
        ComputeSquareWarpPartition(num_warps, M, N, kMPerWarp, kNPerWarp);
  } else {
    TVM_FFI_ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  TVM_FFI_ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)block_size;
    if (op.isWgmma_ || op.isTcgen05_) {
      LOG(FATAL) << "Explicit CUDA GEMM instructions are not available for "
                    "Metal target "
                 << target->str();
    }
    return kMetalSIMDGroup;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    TVM_FFI_ICHECK(gemm_inst == kMetalSIMDGroup)
        << "Unsupported Metal GEMM instruction: " << gemm_inst;
    int num_warps = block_size / TargetMetalGetWarpSize(target);
    return ComputeMetalWarpPartition(policy, M, N, num_warps);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
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
  });
  return true;
}

const bool metal_gemm_registered = RegisterMetalGemm();

} // namespace

} // namespace tl
} // namespace tvm
