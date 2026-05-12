/*!
 * \file tl/backend/rocm/op/gemm.cc
 * \brief ROCm implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "target/utils.h"

#include <cmath>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tir;

namespace rocm {

namespace {

constexpr const char *kROCmMFMA = "rocm.mfma";
constexpr const char *kROCmWMMA = "rocm.wmma";

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                            int num_warps) {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;
  constexpr int kNPerWarp = 16;

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  if (policy.isFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * kMPerWarp) != 0) {
      int max_m_warps = M / kMPerWarp;
      m_warp = max_m_warps;
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (policy.isFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * kNPerWarp) != 0) {
      int max_n_warps = N / kNPerWarp;
      n_warp = max_n_warps;
      m_warp = num_warps / n_warp;
      if (m_warp == 0)
        m_warp = 1;
    }
  } else if (policy.isSquare()) {
    int max_m_warps = M / kMPerWarp;
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= max_m_warps && m <= num_warps; m++) {
      int n = num_warps / m;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      if (m * n != num_warps)
        continue;

      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        best_m = m;
        best_n = n;
      }
    }

    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    (void)op;
    (void)block_size;

    if (TargetIsCDNA(target)) {
      return kROCmMFMA;
    }
    if (TargetIsRDNA(target)) {
      return kROCmWMMA;
    }
    LOG(FATAL) << "Unsupported ROCm target for gemm: " << target->str();
    return kROCmMFMA;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    (void)gemm_inst;
    int num_warps = block_size / TargetGetWarpSize(target);
    return ComputeDefaultWarpPartition(policy, M, N, num_warps);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kROCmMFMA) {
      return "mfma";
    }
    if (gemm_inst == kROCmWMMA) {
      return "wmma";
    }
    return "unknown";
  }
};

} // namespace rocm

namespace {

bool MatchROCmGemmTarget(Target target) { return TargetIsRocm(target); }

bool RegisterROCmGemm() {
  RegisterGemmImpl(GemmImpl{
      "rocm.Gemm",
      MatchROCmGemmTarget,
      rocm::Gemm::SelectInst,
      rocm::Gemm::ComputeWarpPartition,
      rocm::Gemm::ReuseExistingSharedLayout,
      rocm::Gemm::InstructionKind,
  });
  return true;
}

const bool rocm_gemm_registered = RegisterROCmGemm();

} // namespace

} // namespace tl
} // namespace tvm
