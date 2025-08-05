/*!
 * \file tl/op/gemm.cc
 *
 * Define gemm operator.
 */

#include "gemm.h"

#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

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

Gemm::Gemm(Array<PrimExpr> args, BufferMap vmap) {
  Aptr = args[0];
  Bptr = args[1];
  Cptr = args[2];
  A = vmap[GetVarFromAccessPtr(Aptr)];
  B = vmap[GetVarFromAccessPtr(Bptr)];
  C = vmap[GetVarFromAccessPtr(Cptr)];
  trans_A = args[3].as<Bool>().value();
  trans_B = args[4].as<Bool>().value();
  M = args[5].as<IntImm>().value()->value;
  N = args[6].as<IntImm>().value()->value;
  K = args[7].as<IntImm>().value()->value;
  policy = static_cast<GemmWarpPolicy>(args[8].as<IntImm>().value()->value);
  clear_accum = args[9].as<Bool>().value();
  if (args.size() > 10) {
    kPack = args[10].as<IntImm>().value()->value;
    if (kPack != 1 && kPack != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 11) {
    wg_wait = args[11].as<IntImm>().value()->value;
  }
}

Gemm::GemmInst Gemm::GetGemmInst(int block_size, Target target) const {
  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  bool allow_wgmma = TargetIsHopper(target) && (this->M >= 64) &&
                     (num_warps % 4 == 0) && CheckWGMMA();
  if (allow_wgmma) {
    return GemmInst::kWGMMA;
  } else if (TargetIsCDNA(target)) {
    return GemmInst::kMFMA;
  } else if (TargetIsCuda(target)) {
    return GemmInst::kMMA;
  } else {
    ICHECK(0) << "Unsupported target for gemm: " << target->str();
  }
}

std::pair<int, int> Gemm::ComputeWarpPartition(int block_size,
                                               GemmInst gemm_inst,
                                               Target target) const {
  int num_warps = block_size / TargetGetWarpSize(target);
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16; // Rows processed by a single warp
  constexpr int kNPerWarp = 8;  // Columns processed by a single warp

  ICHECK(this->M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << this->M;
  ICHECK(this->N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << this->N;
  if (gemm_inst == GemmInst::kWGMMA) {
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

bool Gemm::CheckWGMMA() const {
  if (C->dtype == DataType::Float(16)) {
    if (A->dtype == DataType::Float(16) && B->dtype == DataType::Float(16))
      return K % 16 == 0;
    else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e4m3())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e5m2())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e4m3())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e5m2())
      return (!trans_A) && trans_B && K % 32 == 0;
    else
      return false;
  } else if (C->dtype == DataType::Float(32)) {
    if (A->dtype == DataType::Float(16) && B->dtype == DataType::Float(16))
      return K % 16 == 0;
    else if (A->dtype == DataType::BFloat(16) &&
             B->dtype == DataType::BFloat(16))
      return K % 16 == 0;
    else if (A->dtype == DataType::Float(32) && B->dtype == DataType::Float(32))
      return (!trans_A) && trans_B && K % 8 == 0;
    else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e4m3())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e5m2())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e4m3())
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e5m2())
      return (!trans_A) && trans_B && K % 32 == 0;
    else
      return false;
  } else if (C->dtype == DataType::Int(32)) {
    if (A->dtype == DataType::Int(8) && B->dtype == DataType::Int(8))
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype == DataType::Int(8) && B->dtype == DataType::UInt(8))
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype == DataType::UInt(8) && B->dtype == DataType::Int(8))
      return (!trans_A) && trans_B && K % 32 == 0;
    else if (A->dtype == DataType::UInt(8) && B->dtype == DataType::UInt(8))
      return (!trans_A) && trans_B && K % 32 == 0;
    else
      return false;
  } else {
    return false;
  }
}

Stmt Gemm::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto block_size = *as_const_int(T.thread_bounds->extent);
  GemmInst gemm_inst = GetGemmInst(block_size, T.target);
  auto [warp_m, warp_n] = ComputeWarpPartition(block_size, gemm_inst, T.target);

  std::stringstream ss;
  std::string op_name = "tl::gemm_ss";
  if (A.scope() == "local.fragment") {
    ICHECK(B.scope() != "local.fragment");
    op_name = "tl::gemm_rs";
  } else if (B.scope() == "local.fragment") {
    op_name = "tl::gemm_sr";
  }
  ss << op_name << "<" << M << ", " << N << ", " << K << ", ";
  ss << warp_m << ", " << warp_n << ", ";
  ss << trans_A << ", " << trans_B;
  ss << ", " << clear_accum;
  if (TargetIsCDNA(T.target)) {
    // for cdna gemm, we need to specify kPack
    ss << ", " << kPack;
  } else if (TargetIsHopper(T.target)) {
    ss << ", " << (gemm_inst == GemmInst::kWGMMA ? "true" : "false");
  }
  if (wg_wait != 0) {
    ss << ", " << wg_wait;
  }
  ss << ">";
  auto A_buffer = T.buffer_remap.count(A) ? T.buffer_remap[A] : A;
  auto B_buffer = T.buffer_remap.count(B) ? T.buffer_remap[B] : B;
  auto C_buffer = T.buffer_remap[C];

  Array<PrimExpr> new_args;
  new_args.push_back(StringImm(ss.str()));
  new_args.push_back(Aptr);
  new_args.push_back(Bptr);
  new_args.push_back(Cptr);
  auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);
  return Evaluate(new_call);
}

LayoutMap Gemm::InferLayout(const LayoutInferArgs &T, InferLevel level) {
  if (completed_)
    return {};
  LayoutMap results;
  ICHECK(C.scope() == "local.fragment");
  auto thread_range = T.thread_bounds;
  auto block_size = *as_const_int(thread_range->extent);
  GemmInst gemm_inst = GetGemmInst(block_size, T.target);
  auto [warp_m, warp_n] = ComputeWarpPartition(block_size, gemm_inst, T.target);

  if (TargetIsVolta(T.target)) {
    auto fragment =
        makeGemmVoltaFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));
    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      results.Set(A, makeGemmVoltaABLayout(*as_const_int(A->shape[dim_A - 2]),
                                           *as_const_int(A->shape[dim_A - 1]),
                                           true, trans_A ? 1 : 2));
    } else if (A.scope() == "local.fragment") {
      ICHECK(trans_A == false);
      auto fragment = makeGemmVoltaFragmentA(M, N, K, M / warp_m, N / warp_n);
      results.Set(A, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }

    ICHECK(B.scope() == "shared" || B.scope() == "shared.dyn");
    int dim_B = B->shape.size();
    results.Set(B, makeGemmVoltaABLayout(*as_const_int(B->shape[dim_B - 2]),
                                         *as_const_int(B->shape[dim_B - 1]),
                                         false, trans_B ? 2 : 1));
  } else if (TargetIsAmpere(T.target) || TargetIsTuring(T.target)) {
    auto fragment =
        makeGemmFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));

    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      const int64_t mat_stride = *as_const_int(A->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(A->shape[dim_A - 1]);
      results.Set(A,
                  makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                   A->dtype.bits(), trans_A ? 1 : 2));
    } else if (A.scope() == "local.fragment") {
      auto fragment = makeGemmFragmentA(M, N, K, M / warp_m, N / warp_n,
                                        A->dtype.bits(), trans_A);
      results.Set(A, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      int dim_B = B->shape.size();
      const int64_t mat_stride = *as_const_int(B->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(B->shape[dim_B - 1]);
      results.Set(B,
                  makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                   B->dtype.bits(), trans_B ? 2 : 1));
    } else if (B.scope() == "local.fragment") {
      auto fragment =
          makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n, trans_B);
      results.Set(B, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else if (TargetIsHopper(T.target)) {
    auto fragment =
        gemm_inst == GemmInst::kWGMMA
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
      auto ABLayout =
          gemm_inst == GemmInst::kWGMMA
              ? makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                       A->dtype.bits(), trans_A ? 1 : 2)
              : makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                 A->dtype.bits(), trans_A ? 1 : 2);
      results.Set(A, ABLayout);
    } else {
      auto fragment = makeGemmFragmentA(M, N, K, M / warp_m, N / warp_n,
                                        A->dtype.bits(), trans_A);
      results.Set(A, fragment->BindThreadRange(thread_range));
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      int dim_B = B->shape.size();
      const int64_t mat_stride = *as_const_int(B->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(B->shape[dim_B - 1]);
      const int64_t continuity =
          trans_B ? mat_continuous : mat_continuous / warp_n;
      auto ABLayout =
          gemm_inst == GemmInst::kWGMMA
              ? makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                       B->dtype.bits(), trans_B ? 2 : 1)
              : makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                 B->dtype.bits(), trans_B ? 2 : 1);
      results.Set(B, ABLayout);
    } else {
      ICHECK(0) << "WGMMA only support B in shared.";
    }
  } else if (TargetIsCDNA(T.target)) {
    auto fragment =
        makeGemmFragmentCCDNA(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));

    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      auto shared_layout = makeGemmABLayoutCDNA(
          *as_const_int(A->shape[dim_A - 2]),
          *as_const_int(A->shape[dim_A - 1]), A->dtype.bits(), kPack);
      results.Set(A, shared_layout);
    } else if (A.scope() == "local.fragment") {
      auto fragment = makeGemmFragmentACDNA(M, N, K, M / warp_m, N / warp_n,
                                            A->dtype.bits(), trans_A);
      results.Set(A, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      int dim_B = B->shape.size();
      auto shared_layout = makeGemmABLayoutCDNA(
          *as_const_int(B->shape[dim_B - 2]),
          *as_const_int(B->shape[dim_B - 1]), B->dtype.bits(), kPack);

      results.Set(B, shared_layout);
    } else if (B.scope() == "local.fragment") {
      auto fragment =
          makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n, trans_B);
      results.Set(B, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else {
    ICHECK(0) << "Not supported " << T.target->str();
  }
  completed_ = true;
  return results;
}

TIR_REGISTER_TL_OP(Gemm, gemm)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace tl
} // namespace tvm