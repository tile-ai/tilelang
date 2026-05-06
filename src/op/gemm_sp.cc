/*!
 * \file tl/op/gemm_sp.cc
 *
 * Define gemm_sp operator.
 */

#include "gemm_sp.h"

#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "utils.h"

#include <vector>

namespace tvm {
namespace tl {

namespace {

std::vector<GemmSPImpl> &GemmSPImplRegistry() {
  static std::vector<GemmSPImpl> registry;
  return registry;
}

const GemmSPImpl &ResolveGemmSPImpl(Target target) {
  const auto &registry = GemmSPImplRegistry();
  const GemmSPImpl *matched_impl = nullptr;
  for (const GemmSPImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.gemm_sp found multiple target-specific implementations for "
          << target->ToDebugString() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.gemm_sp requires a target-specific implementation, but no "
         "gemm_sp implementation is registered for "
      << target->ToDebugString();
  return *matched_impl;
}

} // namespace

void RegisterGemmSPImpl(GemmSPImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.compute_warp_partition != nullptr);
  ICHECK(impl.lower != nullptr);
  ICHECK(impl.infer_layout != nullptr);
  GemmSPImplRegistry().push_back(impl);
}

std::pair<int, int> GemmSPWarpPolicyNode::computeWarpPartition(int M, int N,
                                                               int block_size,
                                                               Target target,
                                                               String gemm_inst,
                                                               int bits) const {
  return ResolveGemmSPImpl(target).compute_warp_partition(
      *this, M, N, block_size, target, gemm_inst, bits);
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
GemmSP::GemmSP(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<GemmSPNode> node = tvm::ffi::make_object<GemmSPNode>();
  auto a_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto e_access = NormalizeToAccessRegion(args[1], kAccessRead);
  auto b_access = NormalizeToAccessRegion(args[2], kAccessRead);
  auto c_access = NormalizeToAccessRegion(args[3], kAccessReadWrite);
  node->aRegion_ = a_access.region;
  node->eRegion_ = e_access.region;
  node->bRegion_ = b_access.region;
  node->cRegion_ = c_access.region;
  node->SetAccessRegions({a_access, e_access, b_access, c_access});
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

AccessRegions GemmSPNode::GetAccessRegions() const {
  AccessRegions result;
  result.reads.push_back(aRegion_);
  result.reads.push_back(eRegion_);
  result.reads.push_back(bRegion_);
  if (!clearAccum_) {
    result.reads.push_back(cRegion_);
  }
  result.writes.push_back(cRegion_);
  return result;
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
 * @brief Lower this GemmSP node through the registered backend.
 *
 * @param T Lowering context containing thread bounds and target.
 * @return Stmt The backend-specific lowered statement.
 */
Stmt GemmSPNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  return ResolveGemmSPImpl(T.target).lower(*this, T, analyzer);
}

/**
 * @brief Infers and returns the memory/layout mapping for the GemmSP operator.
 *
 * Delegates target-specific layout inference to the registered GemmSP backend.
 * The function caches its work: if layout inference has already completed
 * (completed_ == true) it returns an empty LayoutMap.
 *
 * Precondition:
 * - C.scope() must be "local.fragment".
 *
 * @param T LayoutInferArgs containing thread bounds and target.
 * @param level Currently unused inference detail level.
 * @return LayoutMap mapping A, B, and C to their inferred layouts (or empty if
 *         inference was already completed).
 */
LayoutMap GemmSPNode::InferLayout(const LayoutInferArgs &T,
                                  InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results = ResolveGemmSPImpl(T.target).infer_layout(*this, T, level);
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
         String gemm_inst, int bits) {
        policy->computeWarpPartition(M, N, block_size, target, gemm_inst, bits);
        return;
      });
}
} // namespace tl
} // namespace tvm
