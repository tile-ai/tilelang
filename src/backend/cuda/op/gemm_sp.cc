/*!
 * \file tl/backend/cuda/op/gemm_sp.cc
 * \brief CUDA implementation for tl.gemm_sp instruction selection.
 */                                                                            \
#include "op/gemm_sp.h"
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

constexpr const char *kCudaMMASP = "cuda.mma.sp";
constexpr const char *kCudaWGMMASP = "cuda.wgmma.sp";
constexpr const char *kCudaTCGEN05SP = "cuda.tcgen05.sp";

bool CheckWGMMA(const GemmSPNode &op) {
  if (op.B.scope() != "shared.dyn" && op.B.scope() != "shared") {
    return false;
  }

  if (op.C->dtype == DataType::Float(16) ||
      op.C->dtype == DataType::Float(32)) {
    if (op.A->dtype == DataType::Float(16) &&
        op.B->dtype == DataType::Float(16))
      return op.K % 32 == 0;
    else if (op.A->dtype == DataType::BFloat(16) &&
             op.B->dtype == DataType::BFloat(16))
      return op.K % 32 == 0;
    else if (op.A->dtype == DataType::Float(32) &&
             op.B->dtype == DataType::Float(32))
      return (!op.trans_A) && op.trans_B && op.K % 16 == 0;
    else if (op.A->dtype.is_float8() && op.B->dtype.is_float8())
      return (!op.trans_A) && op.trans_B && op.K % 64 == 0;
    else
      return false;
  } else if (op.C->dtype == DataType::Int(32)) {
    if ((op.A->dtype == DataType::Int(8) || op.A->dtype == DataType::UInt(8)) &&
        (op.B->dtype == DataType::Int(8) || op.B->dtype == DataType::UInt(8)))
      return (!op.trans_A) && op.trans_B && op.K % 64 == 0;
    else
      return false;
  } else {
    return false;
  }
}

// TODO @botbw: support tcgen5mma.sp for sparse inputs when it's available
bool AllowTcgen5Mma(const GemmSPNode &op, Target target) {
  bool scope_ok = (IsSharedBuffer(op.A) || op.A.scope() == "shared.tmem") &&
                  IsSharedBuffer(op.B) && op.C.scope() == "shared.tmem";
  if (!TargetIsSm100(target) || !scope_ok)
    return false;
  DataType ab_dtype =
      (op.A.scope() == "shared.tmem") ? op.B->dtype : op.A->dtype;
  return GetTCGEN5MMAMeta(op.M, op.N, op.K, ab_dtype, op.C->dtype).first;
}

bool AllowWgmma(const GemmSPNode &op, int block_size, Target target) {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();

  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  return !ctxt->GetConfig(kDisableWGMMA, Optional<Bool>()).value_or(false) &&
         TargetIsHopper(target) && op.M >= 64 && num_warps % 4 == 0 &&
         CheckWGMMA(op);
}

void FatalWgmmaUnavailable(const GemmSPNode &op, Target target) {
  LOG(FATAL) << "T.wgmma_gemm() requires Hopper WGMMA lowering, but "
                "constraints were not satisfied. Got target="
             << target << ", A(scope=" << op.A.scope()
             << ", dtype=" << op.A->dtype << "), B(scope=" << op.B.scope()
             << ", dtype=" << op.B->dtype << "), C(scope=" << op.C.scope()
             << ", dtype=" << op.C->dtype << "), M=" << op.M << ", N=" << op.N
             << ", K=" << op.K << ".";
}

void FatalTcgen5Unavailable(const GemmSPNode &op, Target target) {
  LOG(FATAL) << "tcgen5";
  //   LOG(FATAL) << "T.tcgen05_gemm() requires Blackwell TCGEN5MMA lowering, "
  //                 "but constraints were not satisfied. Got target="
  //              << target << ", A(scope=" << op.A.scope()
  //              << ", dtype=" << op.A->dtype << "), B(scope=" << op.B.scope()
  //              << ", dtype=" << op.B->dtype << "), C(scope=" << op.C.scope()
  //              << ", dtype=" << op.C->dtype << "), M=" << op.M
  //              << ", N=" << op.N << ", K=" << op.K << ".";
}

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmSPWarpPolicyNode &policy, int M, int N,
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
    ICHECK(0) << "Unknown GemmSPWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

std::pair<int, int>
ComputeWgmmaWarpPartition(const GemmSPWarpPolicyNode &policy, int M, int N,
                          int num_warps) {
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
    ICHECK(0) << "Unknown GemmSPWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct GemmSP {
  static String SelectInst(const GemmSPNode &op, int block_size,
                           Target target) {
    if (op.isWgmma_) {
      if (!AllowWgmma(op, block_size, target)) {
        FatalWgmmaUnavailable(op, target);
      }
      return kCudaWGMMASP;
    }
    if (op.isTcgen05_) {
      FatalTcgen5Unavailable(op, target);

      // if (!AllowTcgen5Mma(op, target)) {
      //     FatalTcgen5Unavailable(op, target);
      // }
      // return kCudaTCGEN05SP;
    }

    if (AllowTcgen5Mma(op, target)) {
      LOG(WARNING) << "TCGEN5MMASP is not yet available for sparse GEMM. "
                      "Falling back to WGMMA or MMA.";
      // return kCudaTCGEN05SP;
    }
    if (AllowWgmma(op, block_size, target)) {
      return kCudaWGMMASP;
    }
    return kCudaMMASP;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmSPWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    int num_warps = block_size / TargetGetWarpSize(target);
    if (gemm_inst == kCudaTCGEN05SP) {
      policy.m_warp = 1;
      policy.n_warp = num_warps;
      return {1, num_warps};
    }
    if (gemm_inst == kCudaWGMMASP) {
      return ComputeWgmmaWarpPartition(policy, M, N, num_warps);
    }
    int k_n_per_warp =
        (TargetIsVolta(target) || TargetIsTuring(target)) ? 16 : 8;
    return ComputeDefaultWarpPartition(policy, M, N, num_warps, k_n_per_warp);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    return gemm_inst == kCudaMMASP;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kCudaWGMMASP) {
      return "wgmma.sp";
    }
    if (gemm_inst == kCudaTCGEN05SP) {
      return "tcgen5mma.sp";
    }
    if (gemm_inst == kCudaMMASP) {
      return "mma.sp";
    }
    return "unknown";
  }
};

} // namespace cuda

namespace {

bool MatchCudaGemmSPTarget(Target target) {
  return TargetIsCuda(target) || TargetIsCuTeDSL(target);
}

bool RegisterCudaGemmSP() {
  RegisterGemmSPImpl(GemmSPImpl{
      "cuda.GemmSP",
      MatchCudaGemmSPTarget,
      cuda::GemmSP::SelectInst,
      cuda::GemmSP::ComputeWarpPartition,
      cuda::GemmSP::ReuseExistingSharedLayout,
      cuda::GemmSP::InstructionKind,
  });
  return true;
}

const bool cuda_gemm_registered = RegisterCudaGemmSP();

} // namespace

// TVM_FFI_STATIC_INIT_BLOCK() {
//   namespace refl = tvm::ffi::reflection;
//   refl::GlobalDef().def(
//       "tl.get_tcgen5_mma_meta", [](int M, int N, int K, DataType ab_dtype,
//                                    DataType c_dtype, bool disable_2cta) {
//         auto [success, meta] =
//             GetTCGEN5MMAMeta(M, N, K, ab_dtype, c_dtype, disable_2cta);
//         Array<Integer> result;
//         if (success) {
//           result.push_back(Integer(meta.atom_m));
//           result.push_back(Integer(meta.atom_n));
//           result.push_back(Integer(meta.atom_k));
//           result.push_back(Integer(meta.enable_ws));
//           result.push_back(Integer(meta.enable_2cta));
//         }
//         return result;
//       });
//   refl::GlobalDef().def(
//       "tl.get_tcgen5_instr_desc",
//       [](int atom_m, int atom_n, int atom_k, DataType ab_dtype,
//          DataType c_dtype, bool a_is_k_major, bool b_is_k_major, int
//          scale_in_a, int scale_in_b) {
//         uint32_t desc = GetTCGEN5InstrDesc(atom_m, atom_n, atom_k, ab_dtype,
//                                            c_dtype, a_is_k_major,
//                                            b_is_k_major, scale_in_a,
//                                            scale_in_b);
//         return Integer(static_cast<int64_t>(desc));
//       });
//   refl::GlobalDef().def("tl.get_tcgen5_blockscaled_instr_desc",
//                         [](int atom_m, int atom_n, DataType ab_dtype,
//                            bool a_is_k_major, bool b_is_k_major, int
//                            scale_in_a, int scale_in_b, int a_sf_id, int
//                            b_sf_id) {
//                           uint32_t desc = GetTCGEN5BlockScaledInstrDesc(
//                               atom_m, atom_n, ab_dtype, a_is_k_major,
//                               b_is_k_major, scale_in_a, scale_in_b, a_sf_id,
//                               b_sf_id);
//                           return Integer(static_cast<int64_t>(desc));
//                         });
// }

} // namespace tl
} // namespace tvm
