/*!
 * \file tl/backend/cuda/op/gemm_sp.cc
 * \brief CUDA implementation for tl.gemm_sp lowering and layout inference.
 */

#include "op/gemm_sp.h"

#include "layout/layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace cuda {

namespace {

constexpr const char *kCudaMMA = "cuda.mma";
constexpr const char *kCudaWGMMA = "cuda.wgmma";

String SelectGemmInst(const GemmSPNode &op, int block_size, Target target) {
  int warp_size = TargetGetWarpSize(target);
  bool maybe_wgmma = TargetIsHopper(target) && op.m_ >= 64 &&
                     (block_size / warp_size % 4 == 0);
  return maybe_wgmma ? String(kCudaWGMMA) : String(kCudaMMA);
}

bool UseWgmma(String gemm_inst) { return gemm_inst == kCudaWGMMA; }

} // namespace

struct GemmSP {
  static std::pair<int, int>
  ComputeWarpPartition(const GemmSPWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst,
                       int bits) {
    int num_warps = block_size / TargetGetWarpSize(target);

    ICHECK(gemm_inst == kCudaMMA || gemm_inst == kCudaWGMMA)
        << "CUDA GemmSP currently only supports MMA and WGMMA";
    auto [m_warp, n_warp] =
        static_cast<const GemmWarpPolicyNode &>(policy).computeWarpPartition(
            M, N, block_size, target, gemm_inst);

    // Special handling for gemm_sp when the tiling size is not a multiple.
    // This should be consistent with shape check in gemm_sp_sm80.h.
    int m_atom_size = bits == 16 ? 32 : 16;
    int n_atom_size = bits == 16 ? 32 : 16;
    static const char *err_msg =
        "Cannot arrange the warp shape to be a multiple of atom size, please "
        "reduce num threads or increase tiling size";
    if (TargetIsAmpere(target)) {
      int warp_shape_m = M / m_warp;
      int warp_shape_n = N / n_warp;
      if (warp_shape_m % m_atom_size) { // GemmWarpPolicy::kFullRow
        m_warp = M / m_atom_size;
        ICHECK(m_warp > 0) << err_msg;
        n_warp = num_warps / m_warp;
        warp_shape_n = N / n_warp;
        ICHECK(warp_shape_n % n_atom_size == 0) << err_msg;
      } else if (warp_shape_n % n_atom_size !=
                 0) { // GemmWarpPolicy::kFullColumn
        n_warp = N / n_atom_size;
        ICHECK(n_warp > 0) << err_msg;
        m_warp = num_warps / n_warp;
        warp_shape_m = M / m_warp;
        ICHECK(warp_shape_m % m_atom_size == 0) << err_msg;
      }
      ICHECK(m_warp * n_warp == num_warps)
          << "m_warp * n_warp must equal num_warps, please report an issue "
             "when encounter this"
          << ", m_warp: " << m_warp << ", n_warp: " << n_warp << ", num_warps"
          << num_warps;
      policy.m_warp = m_warp;
      policy.n_warp = n_warp;
    }
    return {m_warp, n_warp};
  }

  static Stmt Lower(const GemmSPNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    auto block_size = *as_const_int(T.thread_bounds->extent);
    auto gemm_inst = SelectGemmInst(op, block_size, T.target);
    bool maybe_wgmma = UseWgmma(gemm_inst);
    auto [warp_m, warp_n] = op.policy_->computeWarpPartition(
        op.m_, op.n_, block_size, T.target, gemm_inst, op.a_->dtype.bits());

    std::stringstream ss;
    std::string op_name = "tl::gemm_sp_ss";
    ICHECK(IsSharedBuffer(op.a_) && IsSharedBuffer(op.b_))
        << "Only support shared.dyn scope for A and B, but received "
        << op.a_.scope() << " and " << op.b_.scope();
    ICHECK(IsSharedBuffer(op.e_))
        << "Only support shared.dyn scope for E as copy from smem to rmem are "
           "delegated to cute implementation, found "
        << op.e_.scope();
    ss << op_name << "<" << op.m_ << ", " << op.n_ << ", " << op.k_ << ", ";
    ss << warp_m << ", " << warp_n << ", ";
    ss << op.transA_ << ", " << op.transB_;
    ss << ", " << op.clearAccum_;
    if (TargetIsHopper(T.target)) {
      ss << ", " << (maybe_wgmma ? "true" : "false");
    }
    if (op.wgWait_ != 0) {
      ss << ", " << op.wgWait_;
    }
    ss << ">";

    PrimExpr Aptr =
        MakeAccessPtrFromRegion(op.aRegion_, /*r*/ 1, /*require_2d*/ true);
    PrimExpr Bptr =
        MakeAccessPtrFromRegion(op.bRegion_, /*r*/ 1, /*require_2d*/ true);
    PrimExpr Cptr =
        MakeAccessPtrFromRegion(op.cRegion_, /*rw*/ 3, /*require_2d*/ true);
    PrimExpr Eptr =
        MakeAccessPtrFromRegion(op.eRegion_, /*r*/ 1, /*require_2d*/ false);

    auto new_call =
        Call(DataType::Handle(), tl::tl_gemm_sp(),
             Array<PrimExpr>{StringImm(ss.str()), Aptr, Bptr, Cptr, Eptr});
    return Evaluate(new_call);
  }

  static LayoutMap InferLayout(const GemmSPNode &op, const LayoutInferArgs &T,
                               InferLevel level) {
    LayoutMap results;
    ICHECK(IsFragmentBuffer(op.c_));
    auto thread_range = T.thread_bounds;
    auto block_size = *as_const_int(thread_range->extent);
    if (TargetIsHopper(T.target)) {
      auto gemm_inst = SelectGemmInst(op, block_size, T.target);
      bool maybe_wgmma = UseWgmma(gemm_inst);
      auto [warp_m, warp_n] = op.policy_->computeWarpPartition(
          op.m_, op.n_, block_size, T.target, gemm_inst, op.a_->dtype.bits());
      auto fragment =
          maybe_wgmma
              ? makeGemmFragmentCHopper(op.m_, op.n_, op.m_ / warp_m,
                                        op.n_ / warp_n, op.c_->dtype.bits())
              : makeGemmFragmentC(op.m_, op.n_, op.m_ / warp_m, op.n_ / warp_n,
                                  op.c_->dtype.bits());
      results.Set(op.c_, fragment->BindThreadRange(thread_range));
      if (IsSharedBuffer(op.a_)) {
        int dim_A = op.a_->shape.size();
        const int64_t mat_stride = *as_const_int(op.a_->shape[dim_A - 2]);
        const int64_t mat_continuous = *as_const_int(op.a_->shape[dim_A - 1]);
        auto layout =
            makeGemmABLayoutHopper(mat_stride, mat_continuous, mat_continuous,
                                   op.a_->dtype.bits(), op.transA_ ? 1 : 2);
        results.Set(op.a_, ExpandLayoutToMatchBuffer(layout, op.a_));
      } else {
        ICHECK(false) << "Not implemented";
      }

      if (IsSharedBuffer(op.b_)) {
        int dim_B = op.b_->shape.size();
        const int64_t mat_stride = *as_const_int(op.b_->shape[dim_B - 2]);
        const int64_t mat_continuous = *as_const_int(op.b_->shape[dim_B - 1]);
        const int64_t continuity =
            op.transB_ ? mat_continuous : mat_continuous / warp_n;
        auto layout =
            makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                   op.b_->dtype.bits(), op.transB_ ? 2 : 1);
        results.Set(op.b_, ExpandLayoutToMatchBuffer(layout, op.b_));
      } else {
        ICHECK(false) << "WGMMA only support B in shared.";
      }
    } else if (TargetIsAmpere(T.target)) {
      auto [warp_m, warp_n] = op.policy_->computeWarpPartition(
          op.m_, op.n_, block_size, T.target, String(kCudaMMA),
          op.a_->dtype.bits());
      auto fragment = makeGemmSparseFragmentC(
          op.m_, op.n_, op.m_ / warp_m, op.n_ / warp_n, op.c_->dtype.bits());
      results.Set(op.c_, fragment->BindThreadRange(thread_range));

      if (IsSharedBuffer(op.a_)) {
        int dim_A = op.a_->shape.size();
        const int64_t mat_stride = *as_const_int(op.a_->shape[dim_A - 2]);
        const int64_t mat_continuous = *as_const_int(op.a_->shape[dim_A - 1]);
        auto layout = makeGemmSparseAmpereABLayout(mat_stride, mat_continuous,
                                                   op.a_->dtype.bits());
        results.Set(op.a_, ExpandLayoutToMatchBuffer(layout, op.a_));
      } else if (IsFragmentBuffer(op.a_)) {
        ICHECK(false) << "Not Implemented";
      } else {
        ICHECK(0);
      }
      if (IsSharedBuffer(op.b_)) {
        int dim_B = op.b_->shape.size();
        const int64_t mat_stride = *as_const_int(op.b_->shape[dim_B - 2]);
        const int64_t mat_continuous = *as_const_int(op.b_->shape[dim_B - 1]);
        auto layout = makeGemmSparseAmpereABLayout(mat_stride, mat_continuous,
                                                   op.b_->dtype.bits());
        results.Set(op.b_, ExpandLayoutToMatchBuffer(layout, op.b_));
      } else if (IsFragmentBuffer(op.b_)) {
        ICHECK(false) << "Not Implemented";
      } else {
        ICHECK(0);
      }
    } else {
      ICHECK(0) << "Architecture is not supported: " << T.target->str();
    }
    return results;
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
      cuda::GemmSP::ComputeWarpPartition,
      cuda::GemmSP::Lower,
      cuda::GemmSP::InferLayout,
  });
  return true;
}

const bool cuda_gemm_sp_registered = RegisterCudaGemmSP();

} // namespace

} // namespace tl
} // namespace tvm
