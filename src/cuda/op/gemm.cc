/*!
 * \file tl/cuda/op/gemm.cc
 * \brief CUDA implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include "cuda/op/builtin.h"
#include "cuda/target_utils.h"
#include "op/tcgen5_meta.h"
#include "op/utils.h"
#include "span_utils.h"

#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace cuda {

namespace {

constexpr const char *kCudaMMA = "cuda.mma";
constexpr const char *kCudaMMABlockScaled = "cuda.mma.blockscaled";
constexpr const char *kCudaFMA = "cuda.fma";
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
    if (op.a_->dtype.is_tfloat32() && op.b_->dtype.is_tfloat32())
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

  int warp_size = TargetCudaGetWarpSize(target);
  int num_warps = block_size / warp_size;
  return !ctxt->GetConfig(kDisableWGMMA, Optional<Bool>()).value_or(false) &&
         TargetIsHopper(target) && op.m_ >= 64 && num_warps % 4 == 0 &&
         CheckWgmma(op);
}

bool AllowVoltaMma(const GemmNode &op) {
  bool scope_ok = (IsSharedBuffer(op.a_) || IsFragmentBuffer(op.a_)) &&
                  IsSharedBuffer(op.b_);
  if (!scope_ok) {
    return false;
  }
  if (op.transA_) {
    return false;
  }
  if (op.a_->dtype != DataType::Float(16) ||
      op.b_->dtype != DataType::Float(16)) {
    return false;
  }
  if (op.c_->dtype != DataType::Float(16) &&
      op.c_->dtype != DataType::Float(32)) {
    return false;
  }
  return op.m_ % 16 == 0 && op.n_ % 16 == 0 && op.k_ % 4 == 0;
}

bool Use2CtaRequested(const GemmNode &op) {
  if (auto val = op.annotations_.Get("use_2cta")) {
    const auto *imm = val.value().as<IntImmNode>();
    ICHECK(imm) << "use_2cta annotation must be an IntImmNode";
    return imm->value != 0;
  }
  return false;
}

// Native SM75 mma.sync atoms (see tl_templates/cuda/instruction/mma.h).
// Dtype-only on purpose: SM75 MMA handles the same scopes and transposes as
// the generic path, so scope checks here would wrongly demote f16 to FMA.
bool AllowTuringMma(const GemmNode &op) {
  DataType a = op.a_->dtype;
  if (a != op.b_->dtype) {
    return false;
  }
  if (a == DataType::Float(16)) {
    return op.c_->dtype == DataType::Float(16) ||
           op.c_->dtype == DataType::Float(32);
  }
  if ((a.is_int() || a.is_uint()) && (a.bits() == 8 || a.bits() == 4)) {
    return op.c_->dtype == DataType::Int(32);
  }
  return false;
}

bool AllowMmaKShape(const GemmNode &op) {
  int element_bits = op.a_->dtype.bits();
  if (element_bits <= 0 || element_bits > 64 || 256 % element_bits != 0) {
    return false;
  }

  int atom_k = std::min(256 / element_bits, op.k_);
  bool atom_k_supported = atom_k == 4 || atom_k == 8 || atom_k == 16 ||
                          atom_k == 32 || atom_k == 64 || atom_k == 128 ||
                          atom_k == 256;
  return atom_k_supported && op.k_ % atom_k == 0;
}

void FatalWgmmaUnavailable(const GemmNode &op, Target target) {
  LOG(FATAL) << "T.wgmma_gemm() requires Hopper WGMMA lowering, but "
                "constraints were not satisfied. Got target="
             << target << ", A(scope=" << op.a_.scope()
             << ", dtype=" << op.a_->dtype << "), B(scope=" << op.b_.scope()
             << ", dtype=" << op.b_->dtype << "), C(scope=" << op.c_.scope()
             << ", dtype=" << op.c_->dtype << "), M=" << op.m_
             << ", N=" << op.n_ << ", K=" << op.k_ << "."
             << SpanHintSuffix({op.a_->span, op.b_->span, op.c_->span});
}

void FatalTcgen5Unavailable(const GemmNode &op, Target target) {
  LOG(FATAL) << "T.tcgen05_gemm() requires Blackwell TCGEN5MMA lowering, "
                "but constraints were not satisfied. Got target="
             << target << ", A(scope=" << op.a_.scope()
             << ", dtype=" << op.a_->dtype << "), B(scope=" << op.b_.scope()
             << ", dtype=" << op.b_->dtype << "), C(scope=" << op.c_.scope()
             << ", dtype=" << op.c_->dtype << "), M=" << op.m_
             << ", N=" << op.n_ << ", K=" << op.k_ << "."
             << SpanHintSuffix({op.a_->span, op.b_->span, op.c_->span});
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

  auto is_valid = [&](int m, int n) {
    return m * n == num_warps && M % (m * kMPerWarp) == 0 &&
           N % (n * k_n_per_warp) == 0;
  };

  bool found = false;
  if (policy.IsFullRow()) {
    for (int m = num_warps; m >= 1; m--) {
      if (num_warps % m != 0 || !is_valid(m, num_warps / m))
        continue;
      m_warp = m;
      n_warp = num_warps / m;
      found = true;
      break;
    }
  } else if (policy.IsFullCol()) {
    for (int n = num_warps; n >= 1; n--) {
      if (num_warps % n != 0 || !is_valid(num_warps / n, n))
        continue;
      n_warp = n;
      m_warp = num_warps / n;
      found = true;
      break;
    }
  } else if (policy.IsSquare()) {
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= num_warps; m++) {
      if (num_warps % m != 0)
        continue;
      int n = num_warps / m;
      if (!is_valid(m, n))
        continue;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * k_n_per_warp);
      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        m_warp = m;
        n_warp = n;
        found = true;
      }
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  if (!found) {
    LOG(FATAL) << "No valid warp partition for T.gemm: M=" << M << ", N=" << N
               << " cannot be evenly covered by " << num_warps
               << " warps (policy="
               << (policy.IsFullRow()   ? "FullRow"
                   : policy.IsFullCol() ? "FullCol"
                                        : "Square")
               << "). Each warp must own a multiple of " << kMPerWarp
               << " rows and " << k_n_per_warp
               << " columns; adjust `threads` or the block tile shape.";
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

  auto is_valid = [&](int m, int n) {
    return m * n == num_warps && m % kGroup == 0 && M % (m * kMPerWarp) == 0 &&
           N % (n * kNPerWarp) == 0;
  };

  bool found = false;
  if (policy.IsFullRow()) {
    for (int m = num_warps; m >= kGroup; m -= kGroup) {
      if (num_warps % m != 0 || !is_valid(m, num_warps / m))
        continue;
      m_warp = m;
      n_warp = num_warps / m;
      found = true;
      break;
    }
  } else if (policy.IsFullCol()) {
    for (int n = num_warps / kGroup; n >= 1; n--) {
      if (num_warps % n != 0 || !is_valid(num_warps / n, n))
        continue;
      n_warp = n;
      m_warp = num_warps / n;
      found = true;
      break;
    }
  } else if (policy.IsSquare()) {
    float ideal = N > 0 ? static_cast<float>(M) / N : 1.f;

    float best_score = std::numeric_limits<float>::max();
    for (int m = kGroup; m <= num_warps; m += kGroup) {
      if (num_warps % m != 0)
        continue;
      int n = num_warps / m;
      if (!is_valid(m, n))
        continue;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
      float score = std::abs(m_per_warp / n_per_warp - ideal);

      if (score < best_score) {
        best_score = score;
        m_warp = m;
        n_warp = n;
        found = true;
      }
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  if (!found) {
    LOG(FATAL) << "No valid warp partition for T.gemm (Warp-Group MMA): M=" << M
               << ", N=" << N << " cannot be evenly covered by " << num_warps
               << " warps (policy="
               << (policy.IsFullRow()   ? "FullRow"
                   : policy.IsFullCol() ? "FullCol"
                                        : "Square")
               << "). Each warp must own a multiple of " << kMPerWarp
               << " rows and " << kNPerWarp
               << " columns, with m_warp a multiple of " << kGroup
               << "; adjust `threads` or the block tile shape.";
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

    // The public 2CTA shape contract supplies only half of B's N extent per
    // CTA.
    // Falling back to an ordinary one-CTA instruction would therefore be a
    // silent out-of-bounds miscompile rather than a valid fallback.
    if (Use2CtaRequested(op)) {
      if (!AllowTcgen5Mma(op, target)) {
        LOG(FATAL) << "use_2cta=True requires Blackwell TCGEN5MMA "
                      "lowering; no one-CTA instruction fallback is valid.";
      }
      return kCudaTCGEN05;
    }

    if (op.sfaRegion_.defined() || op.sfbRegion_.defined()) {
      if (!op.sfaRegion_.defined() || !op.sfbRegion_.defined()) {
        LOG(FATAL) << "T.mma_gemm_blockscaled() requires both SFA and SFB "
                      "scale-factor regions.";
      }
      if (!TargetIsSM120(target)) {
        LOG(FATAL) << "T.mma_gemm_blockscaled() requires an SM120 CUDA target, "
                      "but got target="
                   << target << "."
                   << SpanHintSuffix({op.a_->span, op.b_->span, op.c_->span});
      }
      return kCudaMMABlockScaled;
    }

    if (AllowTcgen5Mma(op, target)) {
      return kCudaTCGEN05;
    }
    if (AllowWgmma(op, block_size, target)) {
      return kCudaWGMMA;
    }
    if (TargetIsVolta(target) && !AllowVoltaMma(op)) {
      return kCudaFMA;
    }
    if (TargetIsTuring(target) && !AllowTuringMma(op)) {
      return kCudaFMA;
    }
    if (!AllowMmaKShape(op)) {
      return kCudaFMA;
    }
    return kCudaMMA;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    int num_warps = block_size / TargetCudaGetWarpSize(target);
    if (gemm_inst == kCudaTCGEN05) {
      policy.m_warp = 1;
      policy.n_warp = num_warps;
      return {1, num_warps};
    }
    if (gemm_inst == kCudaWGMMA) {
      return ComputeWgmmaWarpPartition(policy, M, N, num_warps);
    }
    if (gemm_inst == kCudaFMA) {
      policy.m_warp = 1;
      policy.n_warp = num_warps;
      return {1, num_warps};
    }
    int k_n_per_warp = TargetIsVolta(target) ? 16 : 8;
    return ComputeDefaultWarpPartition(policy, M, N, num_warps, k_n_per_warp);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    return gemm_inst == kCudaMMA || gemm_inst == kCudaMMABlockScaled;
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
  });
  return true;
}

const bool cuda_gemm_registered = RegisterCudaGemm();

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def(
      "tl.get_tcgen5_mma_meta",
      [](int M, int N, int K, DataType ab_dtype, DataType c_dtype,
         bool disable_2cta, bool disable_ws = false) {
        auto [success, meta] = GetTCGEN5MMAMeta(M, N, K, ab_dtype, c_dtype,
                                                disable_2cta, disable_ws);
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
      [](int atom_m, int atom_n, int atom_k, DataType a_dtype, DataType b_dtype,
         DataType c_dtype, bool a_is_k_major, bool b_is_k_major, int scale_in_a,
         int scale_in_b) {
        uint32_t desc = GetTCGEN5InstrDesc(
            atom_m, atom_n, atom_k, a_dtype, b_dtype, c_dtype, a_is_k_major,
            b_is_k_major, scale_in_a, scale_in_b);
        return Integer(static_cast<int64_t>(desc));
      });
  refl::GlobalDef().def(
      "tl.get_tcgen5_blockscaled_instr_desc",
      [](int atom_m, int atom_n, DataType a_dtype, DataType b_dtype,
         bool a_is_k_major, bool b_is_k_major, int scale_in_a, int scale_in_b,
         int a_sf_id, int b_sf_id) {
        uint32_t desc = GetTCGEN5BlockScaledInstrDesc(
            atom_m, atom_n, a_dtype, b_dtype, a_is_k_major, b_is_k_major,
            scale_in_a, scale_in_b, a_sf_id, b_sf_id);
        return Integer(static_cast<int64_t>(desc));
      });
}

} // namespace tl
} // namespace tvm
