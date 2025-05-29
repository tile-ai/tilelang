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
  B = vmap[GetVarFromAccessPtr(args[1])];
  C = vmap[GetVarFromAccessPtr(args[2])];
  E = vmap[GetVarFromAccessPtr(args[3])];
  trans_A = args[4].as<Bool>().value();
  trans_B = args[5].as<Bool>().value();
  M = args[6].as<IntImm>().value()->value;
  N = args[7].as<IntImm>().value()->value;
  K = args[8].as<IntImm>().value()->value;
  policy = static_cast<GemmWarpPolicy>(args[9].as<IntImm>().value()->value);
  clear_accum = args[10].as<Bool>().value();
  if (args.size() > 11) {
    wg_wait = args[11].as<IntImm>().value()->value;
  }
  if (args.size() > 12) {
    ICHECK(false) << "received " << args.size()
                  << " arguments, but only 10 are expected";
  }
}

std::pair<int, int>
GemmSP::ComputeWarpPartition(int num_warps, Target target,
                             bool maybe_hopper_wgmma) const {
  int m_warp = 1, n_warp = 1;
  bool allow_wgmma = TargetIsHopper(target) && maybe_hopper_wgmma &&
                     (this->M >= 64) && (num_warps % 4 == 0);
  if (allow_wgmma) {
    ICHECK(num_warps % 4 == 0) << "Use Warp Group MMA requires 128*N threads.";
    if (this->policy == GemmWarpPolicy::kFullRow ||
        this->policy == GemmWarpPolicy::kSquare) {
      m_warp = num_warps;
      ICHECK(this->M % num_warps == 0);
    } else if (this->policy == GemmWarpPolicy::kFullCol) {
      m_warp = 4;
      n_warp = num_warps / 4;
      ICHECK(this->N % n_warp == 0);
    } else {
      ICHECK(0) << "Unknown GemmWarpPolicy";
    }
    return {m_warp, n_warp};
  }
  if (this->policy == GemmWarpPolicy::kFullRow) {
    m_warp = num_warps;
    ICHECK(this->M % num_warps == 0);
  } else if (this->policy == GemmWarpPolicy::kFullCol) {
    n_warp = num_warps;
    ICHECK(this->N % num_warps == 0);
  } else if (this->policy == GemmWarpPolicy::kSquare) {
    auto factors = toPrimeFactors(num_warps);
    for (int factor : factors) {
      bool M_divisible = (this->M % (factor * m_warp)) == 0;
      bool N_divisible = (this->N % (factor * n_warp)) == 0;
      if (M_divisible && N_divisible) {
        if (this->M / m_warp >= this->N / n_warp)
          m_warp *= factor;
        else
          n_warp *= factor;
      } else if (M_divisible) {
        m_warp *= factor;
      } else if (N_divisible) {
        n_warp *= factor;
      } else {
        ICHECK(0) << "Cannot compute warp partition for shape" << M << " " << N
                  << " with num_warps " << num_warps;
      }
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }
  // TODO: perform more checks here
  return {m_warp, n_warp};
}

Stmt GemmSP::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  int warp_size = 32;
  if (TargetIsCDNA(T.target)) {
    warp_size = 64;
  }

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
    bool maybe_wgmma = (this->M >= 64) && (block_size / warp_size % 4 == 0);
    ICHECK(maybe_wgmma) << "Only WGMMA is available for now, but disabled "
                           "because  M < 64 or block_size % 128 != 0";
    auto [warp_m, warp_n] =
        ComputeWarpPartition(block_size / warp_size, T.target, maybe_wgmma);
    auto fragment =
        maybe_wgmma
            ? makeGemmFragmentCHopper(M, N, M / warp_m, N / warp_n,
                                      C->dtype.bits())
            : makeGemmFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment);
    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      const int64_t mat_stride = *as_const_int(A->shape[0]);
      const int64_t mat_continuous = *as_const_int(A->shape[1]);
      const int64_t continuity =
          trans_A ? mat_continuous / (warp_m / 4) : mat_continuous;
      results.Set(A, makeGemmABLayout(mat_stride, mat_continuous, continuity,
                                      A->dtype.bits(), trans_A ? 1 : 2));
    } else {
      ICHECK(false) << "Not implemented";
    }

    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      const int64_t mat_stride = *as_const_int(B->shape[0]);
      const int64_t mat_continuous = *as_const_int(B->shape[1]);
      const int64_t continuity =
          trans_B ? mat_continuous : mat_continuous / warp_n;
      results.Set(B, makeGemmABLayout(mat_stride, mat_continuous, continuity,
                                      B->dtype.bits(), trans_B ? 2 : 1));
    } else {
      ICHECK(false) << "WGMMA only support B in shared.";
    }
  } else {
    ICHECK(0) << "Not supported " << T.target->str();
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