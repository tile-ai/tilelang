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

std::pair<int, int> GemmSPWarpPolicyNode::ComputeWarpPartition(int M, int N,
                                                               int block_size,
                                                               Target target,
                                                               bool use_wgmma,
                                                               int bits) const {
  int num_warps = block_size / TargetGetWarpSize(target);

  auto [m_warp, n_warp] = GemmWarpPolicyNode::ComputeWarpPartition(
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
 * @param vmap BufferMap mapping access pointers (from args) to Buffer objects.
 *
 * @note An ICHECK failure is raised if a provided kPack is not 1 or 2.
 */
GemmSP::GemmSP(Array<PrimExpr> args, BufferMap vmap) {
  ObjectPtr<GemmSPNode> node = tvm::ffi::make_object<GemmSPNode>();
  node->A = vmap[GetVarFromAccessPtr(args[0])];
  node->E = vmap[GetVarFromAccessPtr(args[1])];
  node->B = vmap[GetVarFromAccessPtr(args[2])];
  node->C = vmap[GetVarFromAccessPtr(args[3])];
  node->trans_A = args[4].as<Bool>().value();
  node->trans_B = args[5].as<Bool>().value();
  node->M = args[6].as<IntImm>().value()->value;
  node->N = args[7].as<IntImm>().value()->value;
  node->K = args[8].as<IntImm>().value()->value;
  node->policy = GemmSPWarpPolicy(args[9].as<IntImm>().value()->value);
  node->clear_accum = args[10].as<Bool>().value();
  if (args.size() > 11) {
    node->kPack = args[11].as<IntImm>().value()->value;
    if (node->kPack != 1 && node->kPack != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 12) {
    node->wg_wait = args[12].as<IntImm>().value()->value;
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
  bool maybe_wgmma = TargetIsHopper(T.target) && (this->M >= 64) &&
                     (block_size / warp_size % 4 == 0);

  auto [warp_m, warp_n] = policy->ComputeWarpPartition(
      M, N, block_size, T.target, maybe_wgmma, A->dtype.bits());

  std::stringstream ss;
  std::string op_name = "tl::gemm_sp_ss";
  ICHECK((A.scope() == "shared" || A.scope() == "shared.dyn") &&
         (B.scope() == "shared" || B.scope() == "shared.dyn"))
      << "Only support shared.dyn scope for A and B, but received " << A.scope()
      << " and " << B.scope();
  ICHECK((E.scope() == "shared" || E.scope() == "shared.dyn"))
      << "Only support shared.dyn scope for E as copy from smem to rmem are "
         "delegated to cute implementation, found "
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
  ICHECK(C.scope() == "local.fragment");
  auto thread_range = T.thread_bounds;
  auto block_size = *as_const_int(thread_range->extent);
  if (TargetIsHopper(T.target)) {
    const int warp_size = 32;
    constexpr int wgmma_m = 16 * 4;
    bool maybe_wgmma =
        (this->M >= wgmma_m) && (block_size / warp_size % 4 == 0);
    auto [warp_m, warp_n] = policy->ComputeWarpPartition(
        M, N, block_size, T.target, maybe_wgmma, A->dtype.bits());
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
  } else if (TargetIsAmpere(T.target)) {
    auto [warp_m, warp_n] = policy->ComputeWarpPartition(
        M, N, block_size, T.target, false, A->dtype.bits());
    auto fragment =
        makeGemmSparseFragmentC(M, N, M / warp_m, N / warp_n, C->dtype.bits());
    results.Set(C, fragment->BindThreadRange(thread_range));

    if (A.scope() == "shared" || A.scope() == "shared.dyn") {
      int dim_A = A->shape.size();
      const int64_t mat_stride = *as_const_int(A->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(A->shape[dim_A - 1]);
      results.Set(A, makeGemmSparseAmpereABLayout(mat_stride, mat_continuous,
                                                  A->dtype.bits()));
    } else if (A.scope() == "local.fragment") {
      // auto fragment = makeGemmFragmentA(M, N, K, M / warp_m, N / warp_n,
      //                                   A->dtype.bits(), trans_A);
      // results.Set(A, fragment->BindThreadRange(thread_range));
      ICHECK(false) << "Not Implemented";
    } else {
      ICHECK(0);
    }
    if (B.scope() == "shared" || B.scope() == "shared.dyn") {
      int dim_B = B->shape.size();
      const int64_t mat_stride = *as_const_int(B->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(B->shape[dim_B - 1]);
      results.Set(B, makeGemmSparseAmpereABLayout(mat_stride, mat_continuous,
                                                  B->dtype.bits()));
    } else if (B.scope() == "local.fragment") {
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

TIR_REGISTER_TL_OP(GemmSP, gemm_sp)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { GemmSPNode::RegisterReflection(); }

} // namespace tl
} // namespace tvm
