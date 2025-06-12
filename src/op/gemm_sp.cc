// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tl/op/gemm_sp.cc
 *
 * Define gemm_sp operator.
 */

#include "gemm_sp.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

#include "../target/utils.h"
#include "builtin.h"
#include "gemm.h"

namespace tvm {
namespace tl {
static std::vector<int> toPrimeFactors(int x) {
  int i = 2;
  std::vector<int> result;
  while (x > 1) {
    if (x % i == 0) {
      x /= i;
      result.push_back(i);
    } else {
      i++;
    }
  }
  return result;
}

GemmSP::GemmSP(Array<PrimExpr> args, BufferMap vmap) {
  A = vmap[GetVarFromAccessPtr(args[0])];
  E = vmap[GetVarFromAccessPtr(args[1])];
  B = vmap[GetVarFromAccessPtr(args[2])];
  C = vmap[GetVarFromAccessPtr(args[3])];
  trans_A = args[4].as<Bool>().value();
  trans_B = args[5].as<Bool>().value();
  M = args[6].as<IntImm>().value()->value;
  N = args[7].as<IntImm>().value()->value;
  K = args[8].as<IntImm>().value()->value;
  policy = static_cast<GemmWarpPolicy>(args[9].as<IntImm>().value()->value);
  clear_accum = args[10].as<Bool>().value();
  if (args.size() > 11) {
    kPack = args[11].as<IntImm>().value()->value;
    if (kPack != 1 && kPack != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 12) {
    wg_wait = args[12].as<IntImm>().value()->value;
  }
}

std::pair<int, int>
GemmSP::ComputeWarpPartition(int num_warps, Target target,
                             bool maybe_hopper_wgmma) const {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16; // Rows processed by a single warp
  constexpr int kNPerWarp = 8;  // Columns processed by a single warp
  bool allow_wgmma = TargetIsHopper(target) && maybe_hopper_wgmma &&
                     (this->M >= 64) && (num_warps % 4 == 0);
  if (allow_wgmma) {
    ICHECK(num_warps % 4 == 0) << "Warp-Group MMA requires 128×k threads.";

    constexpr int kGroup = 4; // Number of warps in a warp-group

    m_warp = kGroup; // Initially, only one warp-group on M dimension
    n_warp = num_warps / m_warp; // Rest all on N dimension

    if (this->policy == GemmWarpPolicy::kFullRow) {
      // Try to put as many warp-groups as possible on M dimension
      // (decreasing multiples of 4, ensuring divisibility by M)
      for (int cand = num_warps; cand >= kGroup; cand -= kGroup) {
        if (this->M % (cand * kMPerWarp) == 0) {
          m_warp = cand;
          n_warp = num_warps / m_warp;
          break;
        }
      }
    } else if (this->policy == GemmWarpPolicy::kFullCol) {
      // Try to use warps on N dimension; if N is not divisible, split excess
      // groups to M
      int cand_n = n_warp;                       // Initially assume all on N
      if (this->N % (cand_n * kNPerWarp) != 0) { // N direction division fails
        int max_n = this->N / kNPerWarp;
        // Find a feasible n_warp from max possible downwards, ensuring
        // num_warps/n_warp is multiple of 4
        for (int n = std::min(cand_n, max_n); n >= 1; --n) {
          if (num_warps % n == 0 && (num_warps / n) % kGroup == 0) {
            n_warp = n;
            m_warp = num_warps / n_warp;
            break;
          }
        }
      }
    } else if (this->policy == GemmWarpPolicy::kSquare) {
      // Exhaustive search, but m must be multiple of 4
      int max_m = this->M / kMPerWarp;
      int max_n = this->N / kNPerWarp;

      float ideal = this->N > 0 ? static_cast<float>(this->M) / this->N : 1.f;

      float best_score = std::numeric_limits<float>::max();
      int best_m = kGroup, best_n = n_warp;

      for (int m = kGroup; m <= num_warps && m <= max_m; m += kGroup) {
        if (num_warps % m)
          continue;
        int n = num_warps / m;
        if (n > max_n)
          continue;

        float m_per_warp = static_cast<float>(this->M) / (m * kMPerWarp);
        float n_per_warp = static_cast<float>(this->N) / (n * kNPerWarp);
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
        << "m_warp * n_warp must equal num_warps";
    return {m_warp, n_warp};
  }

  if (this->policy == GemmWarpPolicy::kFullRow) {
    // Try to partition M first
    m_warp = num_warps;
    n_warp = 1;

    // If M cannot be evenly divided by m_warp*16, try to split remaining warps
    // to N
    if (this->M % (m_warp * kMPerWarp) != 0) {
      // Calculate how many warps we can use for M
      int max_m_warps = this->M / kMPerWarp;
      m_warp = max_m_warps;
      // Use remaining warps for N
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (this->policy == GemmWarpPolicy::kFullCol) {
    // Try to partition N first
    m_warp = 1;
    n_warp = num_warps;

    // If N cannot be evenly divided by n_warp*8, try to split remaining warps
    // to M
    if (this->N % (n_warp * kNPerWarp) != 0) {
      // Calculate how many warps we can use for N
      int max_n_warps = this->N / kNPerWarp;
      n_warp = max_n_warps;
      // Use remaining warps for M
      m_warp = num_warps / n_warp;
      if (m_warp == 0)
        m_warp = 1;
    }
  } else if (this->policy == GemmWarpPolicy::kSquare) {
    // First calculate the maximum possible warps for each dimension
    int max_m_warps =
        this->M / kMPerWarp; // Each warp needs at least 16 elements in M
    int max_n_warps =
        this->N / kNPerWarp; // Each warp needs at least 8 elements in N

    // Calculate the ideal ratio of M/N warps based on the matrix dimensions
    float ideal_ratio = 1.0f;
    if (this->N > 0) {
      ideal_ratio = static_cast<float>(this->M) / this->N;
    }

    // Start with a balanced initial guess
    m_warp = 1;
    n_warp = 1;

    // Try to find the best balanced partition
    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();

    // Try all possible combinations that satisfy the constraints
    for (int m = 1; m <= max_m_warps && m <= num_warps; m++) {
      int n = num_warps / m;

      // Calculate how balanced this partition is
      float m_per_warp = static_cast<float>(this->M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(this->N) / (n * kNPerWarp);
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
  return {m_warp, n_warp};
}

Stmt GemmSP::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  int warp_size = 32;

  auto block_size = *as_const_int(T.thread_bounds->extent);
  bool maybe_wgmma = TargetIsHopper(T.target) && (this->M >= 64) &&
                     (block_size / warp_size % 4 == 0);

  auto [warp_m, warp_n] =
      ComputeWarpPartition(block_size / warp_size, T.target, maybe_wgmma);

  std::stringstream ss;
  std::string op_name = "tl::gemm_sp_ss";
  ICHECK((A.scope() == "shared" || A.scope() == "shared.dyn") &&
         (B.scope() == "shared" || B.scope() == "shared.dyn"))
      << "Only support shared.dyn scope for A and B, but received " << A.scope()
      << " and " << B.scope();
  ICHECK((E.scope() == "shared" || E.scope() == "shared.dyn"))
      << "Only support shared.dyn scope for E as copy from smem to rmem are "
         "delegated to cute implemntation, found "
      << E.scope();
  ss << op_name << "<" << M << ", " << N << ", " << K << ", ";
  ss << warp_m << ", " << warp_n << ", ";
  ss << trans_A << ", " << trans_B;
  ss << ", " << clear_accum;
  if (TargetIsHopper(T.target)) {
    ss << ", " << (maybe_wgmma ? "true" : "false");
  }
  if (wg_wait != 0) {
    ss << ", " << wg_wait;
  }
  ss << ">";
  auto A_buffer = T.buffer_remap.count(A) ? T.buffer_remap[A] : A;
  auto B_buffer = T.buffer_remap.count(B) ? T.buffer_remap[B] : B;
  auto C_buffer = T.buffer_remap[C];
  auto E_buffer = T.buffer_remap.count(E) ? T.buffer_remap[E] : E;

  Array<PrimExpr> new_args;
  new_args.push_back(StringImm(ss.str()));
  new_args.push_back(A_buffer.access_ptr(1));
  new_args.push_back(B_buffer.access_ptr(1));
  new_args.push_back(C_buffer.access_ptr(3));
  new_args.push_back(E_buffer.access_ptr(1));
  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}

LayoutMap GemmSP::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  if (completed_)
    return {};
  LayoutMap results;
  ICHECK(C.scope() == "local.fragment");
  auto thread_range = T.thread_bounds;
  auto block_size = *as_const_int(thread_range->extent);
  if (TargetIsHopper(T.target)) {
    const int warp_size = 32;
    constexpr int wgmma_m = 16 * 4;
    bool maybe_wgmma =
        (this->M >= wgmma_m) && (block_size / warp_size % 4 == 0);
    auto [warp_m, warp_n] =
        ComputeWarpPartition(block_size / warp_size, T.target, maybe_wgmma);
    auto fragment =
        maybe_wgmma
            ? makeGemmFragmentCHopper(M, N, M / warp_m, N / warp_n,
                                      C->dtype.bits())
            : makeGemmFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));
    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      const int64_t mat_stride = *as_const_int(A->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(A->shape[dim_A - 1]);
      const int64_t continuity =
          trans_A ? 4 * mat_continuous / warp_m : mat_continuous;
      results.Set(A, makeGemmABLayoutHopper(mat_stride, mat_continuous,
                                            mat_continuous, A->dtype.bits(),
                                            trans_A ? 1 : 2));
    } else {
      ICHECK(false) << "Not implemented";
    }

    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      int dim_B = B->shape.size();
      const int64_t mat_stride = *as_const_int(B->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(B->shape[dim_B - 1]);
      const int64_t continuity =
          trans_B ? mat_continuous : mat_continuous / warp_n;
      results.Set(B,
                  makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                         B->dtype.bits(), trans_B ? 2 : 1));
    } else {
      ICHECK(false) << "WGMMA only support B in shared.";
    }
  } else {
    ICHECK(0) << "Not supported " << T.target->str()
              << " Currently only Hopper are supported";
  }
  completed_ = true;
  return results;
}
TIR_REGISTER_TL_OP(GemmSP, gemm_sp)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm
