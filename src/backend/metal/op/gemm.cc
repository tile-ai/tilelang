/*!
 * \file tl/backend/metal/op/gemm.cc
 * \brief Metal implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "backend/metal/op/utils.h"
#include "target/utils.h"

#include <tvm/runtime/logging.h>

#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

namespace {

constexpr const char *kMetalSIMDGroup = "metal.simdgroup";
constexpr const char *kMetalCooperativeTensor = "metal.cooperative_tensor";

std::pair<int, int> ComputeMetalWarpPartition(const GemmWarpPolicyNode &policy,
                                              int M, int N, int num_warps,
                                              String gemm_inst) {
  int m_warp = 1, n_warp = 1;
  int kMPerWarp, kNPerWarp;
  if (gemm_inst == kMetalCooperativeTensor) {
    kMPerWarp = 16;
    kNPerWarp = 32;
  } else {
    // kMetalSIMDGroup: keep existing 8x8 micro tile
    kMPerWarp = 8;
    kNPerWarp = 8;
  }

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

bool CanUseCooperativeTensor(const GemmWarpPolicyNode &policy, int M, int N,
                             int K, int num_warps) {
  constexpr int kMPerWarp = 16;
  constexpr int kNPerWarp = 32;
  if (M % kMPerWarp != 0 || N % kNPerWarp != 0 || K % 16 != 0) {
    return false;
  }
  int max_m = M / kMPerWarp;
  int max_n = N / kNPerWarp;
  if (policy.isFullRow()) {
    int m_warp = num_warps;
    if (M % (m_warp * kMPerWarp) != 0) {
      m_warp = max_m;
    }
    return m_warp > 0 && num_warps % m_warp == 0 && num_warps / m_warp <= max_n;
  }
  if (policy.isFullCol()) {
    int n_warp = num_warps;
    if (N % (n_warp * kNPerWarp) != 0) {
      n_warp = max_n;
    }
    return n_warp > 0 && num_warps % n_warp == 0 && num_warps / n_warp <= max_m;
  }
  if (policy.isSquare()) {
    for (int m = 1; m <= std::min(num_warps, max_m); ++m) {
      if (num_warps % m == 0 && num_warps / m <= max_n) {
        return true;
      }
    }
  }
  return false;
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
    if (op.c_.scope() == "local.fragment" ||
        op.c_.scope() == "metal.simdgroup") {
      return kMetalSIMDGroup;
    }
    int num_warps = block_size / TargetGetWarpSize(target);
    if (CanUseCooperativeTensor(*op.policy_.operator->(), op.m_, op.n_, op.k_,
                                num_warps)) {
      return kMetalCooperativeTensor;
    }
    return kMetalSIMDGroup;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    TVM_FFI_ICHECK(gemm_inst == kMetalSIMDGroup ||
                   gemm_inst == kMetalCooperativeTensor)
        << "Unsupported Metal GEMM instruction: " << gemm_inst;
    int num_warps = block_size / TargetGetWarpSize(target);
    return ComputeMetalWarpPartition(policy, M, N, num_warps, gemm_inst);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kMetalSIMDGroup) {
      return "metal_simdgroup";
    }
    if (gemm_inst == kMetalCooperativeTensor) {
      return "metal_cooperative_tensor";
    }
    return "unknown";
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
