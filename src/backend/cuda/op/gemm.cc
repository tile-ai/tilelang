/*!
 * \file tl/backend/cuda/op/gemm.cc
 * \brief CUDA implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "op/builtin.h"
#include "op/tcgen5_meta.h"
#include "op/utils.h"
#include "target/utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tir;

namespace cuda {

namespace {

constexpr const char *kCudaMMA = "cuda.mma";
constexpr const char *kCudaWGMMA = "cuda.wgmma";
constexpr const char *kCudaTCGEN05 = "cuda.tcgen05";

bool CheckWgmma(const GemmNode &op) {
  if (op.b_.scope() != "shared.dyn" && op.b_.scope() != "shared") {
    return false;
  }

  if (op.c_->dtype == DataType::Float(16)) {
    if (op.a_->dtype == DataType::Float(16) &&
        op.b_->dtype == DataType::Float(16))
      return op.k_ % 16 == 0;
    if (op.a_->dtype.is_float8() && op.b_->dtype.is_float8())
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    return false;
  }
  if (op.c_->dtype == DataType::Float(32)) {
    if (op.a_->dtype == DataType::Float(16) &&
        op.b_->dtype == DataType::Float(16))
      return op.k_ % 16 == 0;
    if (op.a_->dtype == DataType::BFloat(16) &&
        op.b_->dtype == DataType::BFloat(16))
      return op.k_ % 16 == 0;
    if (op.a_->dtype == DataType::Float(32) &&
        op.b_->dtype == DataType::Float(32))
      return (!op.transA_) && op.transB_ && op.k_ % 8 == 0;
    if (op.a_->dtype.is_float8() && op.b_->dtype.is_float8())
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    return false;
  }
  if (op.c_->dtype == DataType::Int(32)) {
    if (op.a_->dtype == DataType::Int(8) && op.b_->dtype == DataType::Int(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    if (op.a_->dtype == DataType::Int(8) && op.b_->dtype == DataType::UInt(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    if (op.a_->dtype == DataType::UInt(8) && op.b_->dtype == DataType::Int(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    if (op.a_->dtype == DataType::UInt(8) && op.b_->dtype == DataType::UInt(8))
      return (!op.transA_) && op.transB_ && op.k_ % 32 == 0;
    return false;
  }
  return false;
}

bool AllowTcgen5Mma(const GemmNode &op, Target target) {
  bool scope_ok = (IsSharedBuffer(op.a_) || op.a_.scope() == "shared.tmem") &&
                  IsSharedBuffer(op.b_) && op.c_.scope() == "shared.tmem";
  if (!TargetIsSm100(target) || !scope_ok)
    return false;
  DataType ab_dtype =
      (op.a_.scope() == "shared.tmem") ? op.b_->dtype : op.a_->dtype;
  return GetTCGEN5MMAMeta(op.m_, op.n_, op.k_, ab_dtype, op.c_->dtype).first;
}

bool AllowWgmma(const GemmNode &op, int block_size, Target target) {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();

  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  return !ctxt->GetConfig(kDisableWGMMA, Optional<Bool>()).value_or(false) &&
         TargetIsHopper(target) && op.m_ >= 64 && num_warps % 4 == 0 &&
         CheckWgmma(op);
}

void FatalWgmmaUnavailable(const GemmNode &op, Target target) {
  LOG(FATAL) << "T.wgmma_gemm() requires Hopper WGMMA lowering, but "
                "constraints were not satisfied. Got target="
             << target << ", A(scope=" << op.a_.scope()
             << ", dtype=" << op.a_->dtype << "), B(scope=" << op.b_.scope()
             << ", dtype=" << op.b_->dtype << "), C(scope=" << op.c_.scope()
             << ", dtype=" << op.c_->dtype << "), M=" << op.m_
             << ", N=" << op.n_ << ", K=" << op.k_ << ".";
}

void FatalTcgen5Unavailable(const GemmNode &op, Target target) {
  LOG(FATAL) << "T.tcgen05_gemm() requires Blackwell TCGEN5MMA lowering, "
                "but constraints were not satisfied. Got target="
             << target << ", A(scope=" << op.a_.scope()
             << ", dtype=" << op.a_->dtype << "), B(scope=" << op.b_.scope()
             << ", dtype=" << op.b_->dtype << "), C(scope=" << op.c_.scope()
             << ", dtype=" << op.c_->dtype << "), M=" << op.m_
             << ", N=" << op.n_ << ", K=" << op.k_ << ".";
}

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                            int num_warps, int k_n_per_warp) {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % k_n_per_warp == 0)
      << "N must be divisible by " << k_n_per_warp << ", but got " << N;

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
    if (N % (n_warp * k_n_per_warp) != 0) {
      int max_n_warps = N / k_n_per_warp;
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
      float n_per_warp = static_cast<float>(N) / (n * k_n_per_warp);
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

std::pair<int, int> ComputeWgmmaWarpPartition(const GemmWarpPolicyNode &policy,
                                              int M, int N, int num_warps) {
  ICHECK(num_warps % 4 == 0) << "Warp-Group MMA requires 128*k threads.";

  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;
  constexpr int kNPerWarp = 8;
  constexpr int kGroup = 4;

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  m_warp = kGroup;
  n_warp = num_warps / m_warp;

  if (policy.isFullRow()) {
    for (int cand = num_warps; cand >= kGroup; cand -= kGroup) {
      if (M % (cand * kMPerWarp) == 0) {
        m_warp = cand;
        n_warp = num_warps / m_warp;
        break;
      }
    }
  } else if (policy.isFullCol()) {
    int cand_n = n_warp;
    if (N % (cand_n * kNPerWarp) != 0) {
      int max_n = N / kNPerWarp;
      for (int n = std::min(cand_n, max_n); n >= 1; --n) {
        if (num_warps % n == 0 && (num_warps / n) % kGroup == 0) {
          n_warp = n;
          m_warp = num_warps / n_warp;
          break;
        }
      }
    }
  } else if (policy.isSquare()) {
    int max_m = M / kMPerWarp;
    int max_n = N / kNPerWarp;

    float ideal = N > 0 ? static_cast<float>(M) / N : 1.f;
    float best_score = std::numeric_limits<float>::max();
    int best_m = kGroup, best_n = n_warp;

    for (int m = kGroup; m <= num_warps && m <= max_m; m += kGroup) {
      if (num_warps % m)
        continue;
      int n = num_warps / m;
      if (n > max_n)
        continue;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      float score = std::abs(m_per_warp / n_per_warp - ideal);

      if (score < best_score) {
        best_score = score;
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
    if (op.isWgmma_) {
      if (!AllowWgmma(op, block_size, target)) {
        FatalWgmmaUnavailable(op, target);
      }
      return kCudaWGMMA;
    }
    if (op.isTcgen05_) {
      if (!AllowTcgen5Mma(op, target)) {
        FatalTcgen5Unavailable(op, target);
      }
      return kCudaTCGEN05;
    }

    if (AllowTcgen5Mma(op, target)) {
      return kCudaTCGEN05;
    }
    if (AllowWgmma(op, block_size, target)) {
      return kCudaWGMMA;
    }
    return kCudaMMA;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    int num_warps = block_size / TargetGetWarpSize(target);
    if (gemm_inst == kCudaTCGEN05) {
      policy.m_warp = 1;
      policy.n_warp = num_warps;
      return {1, num_warps};
    }
    if (gemm_inst == kCudaWGMMA) {
      return ComputeWgmmaWarpPartition(policy, M, N, num_warps);
    }
    int k_n_per_warp = TargetIsVolta(target) ? 16 : 8;
    return ComputeDefaultWarpPartition(policy, M, N, num_warps, k_n_per_warp);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    return gemm_inst == kCudaMMA;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kCudaWGMMA) {
      return "wgmma";
    }
    if (gemm_inst == kCudaTCGEN05) {
      return "tcgen5mma";
    }
    if (gemm_inst == kCudaMMA) {
      return "mma";
    }
    return "unknown";
  }
};

} // namespace cuda

namespace {

bool MatchCudaGemmTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaGemm() {
  RegisterGemmImpl(GemmImpl{
      "cuda.Gemm",
      MatchCudaGemmTarget,
      cuda::Gemm::SelectInst,
      cuda::Gemm::ComputeWarpPartition,
      cuda::Gemm::ReuseExistingSharedLayout,
      cuda::Gemm::InstructionKind,
  });
  return true;
}

const bool cuda_gemm_registered = RegisterCudaGemm();

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.get_tcgen5_mma_meta", [](int M, int N, int K, DataType ab_dtype,
                                   DataType c_dtype, bool disable_2cta) {
        auto [success, meta] =
            GetTCGEN5MMAMeta(M, N, K, ab_dtype, c_dtype, disable_2cta);
        Array<Integer> result;
        if (success) {
          result.push_back(Integer(meta.atom_m));
          result.push_back(Integer(meta.atom_n));
          result.push_back(Integer(meta.atom_k));
          result.push_back(Integer(meta.enable_ws));
          result.push_back(Integer(meta.enable_2cta));
        }
        return result;
      });
  refl::GlobalDef().def(
      "tl.get_tcgen5_instr_desc",
      [](int atom_m, int atom_n, int atom_k, DataType ab_dtype,
         DataType c_dtype, bool a_is_k_major, bool b_is_k_major, int scale_in_a,
         int scale_in_b) {
        uint32_t desc = GetTCGEN5InstrDesc(atom_m, atom_n, atom_k, ab_dtype,
                                           c_dtype, a_is_k_major, b_is_k_major,
                                           scale_in_a, scale_in_b);
        return Integer(static_cast<int64_t>(desc));
      });
  refl::GlobalDef().def("tl.get_tcgen5_blockscaled_instr_desc",
                        [](int atom_m, int atom_n, DataType ab_dtype,
                           bool a_is_k_major, bool b_is_k_major, int scale_in_a,
                           int scale_in_b, int a_sf_id, int b_sf_id) {
                          uint32_t desc = GetTCGEN5BlockScaledInstrDesc(
                              atom_m, atom_n, ab_dtype, a_is_k_major,
                              b_is_k_major, scale_in_a, scale_in_b, a_sf_id,
                              b_sf_id);
                          return Integer(static_cast<int64_t>(desc));
                        });
}

} // namespace tl
} // namespace tvm
