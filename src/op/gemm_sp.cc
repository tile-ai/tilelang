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
#include "utils.h"

namespace tvm {
namespace tl {

std::pair<int, int> GemmSPWarpPolicyNode::computeWarpPartition(int M, int N,
                                                               int block_size,
                                                               Target target,
                                                               bool use_wgmma,
                                                               int bits) const {
  int num_warps = block_size / TargetGetWarpSize(target);

  auto [m_warp, n_warp] = GemmWarpPolicyNode::computeWarpPartition(
      M, N, block_size, target, use_wgmma ? GemmInst::kWGMMA : GemmInst::kMMA);

  // Special handling for gemm_sp when the tiling size is not a multiple
  // This should be consistent with shape check in gemm_sp_sm80.h
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
    } else if (warp_shape_n % n_atom_size != 0) { // GemmWarpPolicy::kFullColumn
      n_warp = N / n_atom_size;
      ICHECK(n_warp > 0) << err_msg;
      m_warp = num_warps / n_warp;
      warp_shape_m = M / m_warp;
      ICHECK(warp_shape_m % m_atom_size == 0) << err_msg;
    }
    ICHECK(m_warp * n_warp == num_warps)
        << "m_warp * n_warp must equal num_warps, please report an issue when "
           "encounter this"
        << ", m_warp: " << m_warp << ", n_warp: " << n_warp << ", num_warps"
        << num_warps;
    this->m_warp = m_warp;
    this->n_warp = n_warp;
  }
  return {m_warp, n_warp};
}

/**
 * @brief Construct a GemmSP operator node from TL call arguments and a buffer
 * map.
 *
 * Parses the expected call argument tuple and fills an internal GemmSPNode:
 * - Buffers: A (args[0]), E (args[1]), B (args[2]), C (args[3]) are looked up
 * in vmap.
 * - Booleans: trans_A (args[4]), trans_B (args[5]).
 * - Dimensions: M (args[6]), N (args[7]), K (args[8]) as integers.
 * - Warp policy: policy (args[9]) mapped to GemmWarpPolicy.
 * - clear_accum: boolean flag (args[10]).
 * - Optional kPack (args[11]): must be 1 or 2 (checked via ICHECK).
 * - Optional wg_wait (args[12]): integer workgroup wait parameter.
 *
 * The populated GemmSPNode is stored in the instance's internal data_ pointer.
 *
 * @param args Positional TL call arguments in the above order.
 *
 * @note An ICHECK failure is raised if a provided kPack is not 1 or 2.
 */
GemmSP::GemmSP(Array<PrimExpr> args) {
  ObjectPtr<GemmSPNode> node = tvm::ffi::make_object<GemmSPNode>();
  node->aRegion_ = NormalizeToBufferRegion(args[0]);
  node->eRegion_ = NormalizeToBufferRegion(args[1]);
  node->bRegion_ = NormalizeToBufferRegion(args[2]);
  node->cRegion_ = NormalizeToBufferRegion(args[3]);
  node->a_ = node->aRegion_->buffer;
  node->e_ = node->eRegion_->buffer;
  node->b_ = node->bRegion_->buffer;
  node->c_ = node->cRegion_->buffer;
  node->transA_ = args[4].as<Bool>().value();
  node->transB_ = args[5].as<Bool>().value();
  node->m_ = args[6].as<IntImm>().value()->value;
  node->n_ = args[7].as<IntImm>().value()->value;
  node->k_ = args[8].as<IntImm>().value()->value;
  node->policy_ = GemmSPWarpPolicy(args[9].as<IntImm>().value()->value);
  node->clearAccum_ = args[10].as<Bool>().value();
  if (args.size() > 11) {
    node->kPack_ = args[11].as<IntImm>().value()->value;
    if (node->kPack_ != 1 && node->kPack_ != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 12) {
    node->wgWait_ = args[12].as<IntImm>().value()->value;
  }
  data_ = std::move(node);
}

/**
 * @brief Create a deep copy of this GemmSPNode wrapped as a TileOperator.
 *
 * Returns a new TileOperator that owns a copy of this node. The cloned node
 * duplicates all fields of the original; subsequent modifications to the
 * clone do not affect the original node.
 *
 * @return TileOperator A TileOperator holding a cloned GemmSPNode.
 */
TileOperator GemmSPNode::Clone() const {
  auto op = tvm::ffi::make_object<GemmSPNode>(*this);
  return GemmSP(op);
}

/**
 * @brief Lower this GemmSP node to a TL (tensile-like) intrinsic call.
 *
 * Constructs and returns an Evaluate statement containing a call to the
 * TL gemm_sp intrinsic that encodes this GEMM's template parameters
 * (M, N, K, warp partition, transposition flags, clear_accum, and optional
 * Hopper/WGMMA and wg_wait modifiers) and the remapped buffer access pointers.
 *
 * The function validates that A, B, and E reside in shared (or shared.dyn)
 * memory (ICHECK failures otherwise), computes the warp partition based on
 * the launch configuration and target, and emits a single tl::tl_gemm_sp call
 * with a string template describing the configuration.
 *
 * @param T Lowering context containing thread bounds, target, and optional
 *          buffer remapping used to obtain the final buffer AccessPtr
 *          arguments for the TL call.
 * @return Stmt An Evaluate wrapping the constructed tl::tl_gemm_sp call.
 */
Stmt GemmSPNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  int warp_size = 32;

  auto block_size = *as_const_int(T.thread_bounds->extent);
  bool maybe_wgmma = TargetIsHopper(T.target) && (this->m_ >= 64) &&
                     (block_size / warp_size % 4 == 0);

  auto [warp_m, warp_n] = policy_->computeWarpPartition(
      m_, n_, block_size, T.target, maybe_wgmma, a_->dtype.bits());

  std::stringstream ss;
  std::string op_name = "tl::gemm_sp_ss";
  ICHECK((a_.scope() == "shared" || a_.scope() == "shared.dyn") &&
         (b_.scope() == "shared" || b_.scope() == "shared.dyn"))
      << "Only support shared.dyn scope for A and B, but received "
      << a_.scope() << " and " << b_.scope();
  ICHECK((e_.scope() == "shared" || e_.scope() == "shared.dyn"))
      << "Only support shared.dyn scope for E as copy from smem to rmem are "
         "delegated to cute implementation, found "
      << e_.scope();
  ss << op_name << "<" << m_ << ", " << n_ << ", " << k_ << ", ";
  ss << warp_m << ", " << warp_n << ", ";
  ss << transA_ << ", " << transB_;
  ss << ", " << clearAccum_;
  if (TargetIsHopper(T.target)) {
    ss << ", " << (maybe_wgmma ? "true" : "false");
  }
  if (wgWait_ != 0) {
    ss << ", " << wgWait_;
  }
  ss << ">";
  auto A_buffer = T.buffer_remap.count(a_) ? T.buffer_remap[a_] : a_;
  auto B_buffer = T.buffer_remap.count(b_) ? T.buffer_remap[b_] : b_;
  auto C_buffer = T.buffer_remap[c_];
  auto E_buffer = T.buffer_remap.count(e_) ? T.buffer_remap[e_] : e_;

  auto new_call =
      Call(DataType::Handle(), tl::tl_gemm_sp(),
           Array<PrimExpr>{StringImm(ss.str()), A_buffer.access_ptr(1),
                           B_buffer.access_ptr(1), C_buffer.access_ptr(3),
                           E_buffer.access_ptr(1)});
  return Evaluate(new_call);
}

/**
 * @brief Infers and returns the memory/layout mapping for the GemmSP operator.
 *
 * Infers thread-local fragment layout for C and shared-memory layouts for A and
 * B based on the target (Hopper-only path), block/thread bounds in T,
 * transposition flags, and matrix dimensions stored in the node. The function
 * caches its work: if layout inference has already completed (completed_ ==
 * true) it returns an empty LayoutMap.
 *
 * Precondition:
 * - C.scope() must be "local.fragment".
 *
 * Behavior notes:
 * - Only the Hopper target is supported; non-Hopper targets trigger a fatal
 * check.
 * - For Hopper, the function computes a warp partition from block size and may
 *   enable WGMMA-specific fragment creation when conditions on M and block size
 *   are met.
 * - A and B must reside in "shared" or "shared.dyn"; otherwise the function
 *   aborts with a check failure.
 * - The method sets completed_ = true before returning to avoid re-entrance.
 *
 * @param T LayoutInferArgs containing thread bounds and the target (used to
 *          select Hopper-specific layouts).
 * @param level Currently unused inference detail level.
 * @return LayoutMap mapping A, B, and C to their inferred layouts (or empty if
 *         inference was already completed).
 */
LayoutMap GemmSPNode::InferLayout(const LayoutInferArgs &T,
                                  InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  ICHECK(IsFragmentBuffer(c_));
  auto thread_range = T.thread_bounds;
  auto block_size = *as_const_int(thread_range->extent);
  if (TargetIsHopper(T.target)) {
    const int warp_size = 32;
    constexpr int wgmma_m = 16 * 4;
    bool maybe_wgmma =
        (this->m_ >= wgmma_m) && (block_size / warp_size % 4 == 0);
    auto [warp_m, warp_n] = policy_->computeWarpPartition(
        m_, n_, block_size, T.target, maybe_wgmma, a_->dtype.bits());
    auto fragment = maybe_wgmma
                        ? makeGemmFragmentCHopper(m_, n_, m_ / warp_m,
                                                  n_ / warp_n, c_->dtype.bits())
                        : makeGemmFragmentC(m_, n_, m_ / warp_m, n_ / warp_n,
                                            c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));
    if (a_.scope() == "shared" || a_.scope() == "shared.dyn") {
      int dim_A = a_->shape.size();
      const int64_t mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      results.Set(a_, makeGemmABLayoutHopper(mat_stride, mat_continuous,
                                             mat_continuous, a_->dtype.bits(),
                                             transA_ ? 1 : 2));
    } else {
      ICHECK(false) << "Not implemented";
    }

    if (b_.scope() == "shared" || b_.scope() == "shared.dyn") {
      int dim_B = b_->shape.size();
      const int64_t mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      const int64_t continuity =
          transB_ ? mat_continuous : mat_continuous / warp_n;
      results.Set(b_,
                  makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                         b_->dtype.bits(), transB_ ? 2 : 1));
    } else {
      ICHECK(false) << "WGMMA only support B in shared.";
    }
  } else if (TargetIsAmpere(T.target)) {
    auto [warp_m, warp_n] = policy_->computeWarpPartition(
        m_, n_, block_size, T.target, false, a_->dtype.bits());
    auto fragment = makeGemmSparseFragmentC(m_, n_, m_ / warp_m, n_ / warp_n,
                                            c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));

    if (a_.scope() == "shared" || a_.scope() == "shared.dyn") {
      int dim_A = a_->shape.size();
      const int64_t mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      results.Set(a_, makeGemmSparseAmpereABLayout(mat_stride, mat_continuous,
                                                   a_->dtype.bits()));
    } else if (IsFragmentBuffer(a_)) {
      // auto fragment = makeGemmFragmentA(M, N, K, M / warp_m, N / warp_n,
      //                                   A->dtype.bits(), trans_A);
      // results.Set(A, fragment->BindThreadRange(thread_range));
      ICHECK(false) << "Not Implemented";
    } else {
      ICHECK(0);
    }
    if (b_.scope() == "shared" || b_.scope() == "shared.dyn") {
      int dim_B = b_->shape.size();
      const int64_t mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      results.Set(b_, makeGemmSparseAmpereABLayout(mat_stride, mat_continuous,
                                                   b_->dtype.bits()));
    } else if (IsFragmentBuffer(b_)) {
      // auto fragment =
      //     makeGemmFragmentB(M, N, K, M / warp_m, N / warp_n, trans_B);
      // results.Set(B, fragment->BindThreadRange(thread_range));
      ICHECK(false) << "Not Implemented";
    } else {
      ICHECK(0);
    }
  } else {
    ICHECK(0) << "Architecture is not supported: " << T.target->str();
  }
  completed_ = true;
  return results;
}

TIR_REGISTER_TL_TILE_OP(GemmSP, gemm_sp)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.GemmSPWarpPolicy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "GemmSPWarpPolicy");

TVM_FFI_STATIC_INIT_BLOCK() {
  GemmSPNode::RegisterReflection();
  GemmSPWarpPolicyNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.GemmSPWarpPolicyComputeWarpPartition",
      [](GemmSPWarpPolicy policy, int M, int N, int block_size, Target target,
         bool use_wgmma, int bits) {
        policy->computeWarpPartition(M, N, block_size, target, use_wgmma, bits);
        return;
      });
}
} // namespace tl
} // namespace tvm
