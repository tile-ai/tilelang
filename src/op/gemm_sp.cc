/*!
 * \file tl/op/gemm_sp.cc
 *
 * Define gemm_sp operator.
 */

#include "gemm_sp.h"
#include "layout/expr_utils.h"
#include "support/check.h"
#include "utils.h"

#include "builtin.h"
#include "lowering.h"
#include <tvm/ir/prim/op.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/math.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>

#include "backend/common/target_utils.h"
#include "tvm/ffi/string.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
using namespace ffi;

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
          << target->str() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.gemm_sp requires a target-specific implementation, but no "
         "gemm_sp implementation is registered for "
      << target->str();
  return *matched_impl;
}

} // namespace

std::pair<int, int> GemmSPWarpPolicyNode::ComputeWarpPartition(
    int M, int N, int block_size, Target target, String gemm_inst) const {
  return ResolveGemmSPImpl(target).compute_warp_partition(
      *this, M, N, block_size, target, gemm_inst);
}

void RegisterGemmSPImpl(GemmSPImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.select_inst != nullptr);
  ICHECK(impl.compute_warp_partition != nullptr);
  ICHECK(impl.reuse_existing_shared_layout != nullptr);
  GemmSPImplRegistry().push_back(impl);
}

/**
 * @brief Construct a GemmSP operator from serialized TL arguments.
 *
 * Deserializes operator parameters from `args` and resolves buffer references,
 * populating an internal GemmSPNode with buffers, transpose flags, M/N/K,
 * warp policy and clear_accum.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Eptr, Bptr, Cptr, trans_A (Bool), trans_E (Bool),
 *      trans_B (Bool), M (Int), N (Int), K (Int), policy (Int),
 *      clear_accum (Bool)]
 *   Backend lowering knobs (wg_wait) ride in the annotations map.
 */
GemmSP::GemmSP(Array<Expr> args, Map<String, Any> annotations) {
  ICHECK_EQ(args.size(), 12U);
  ObjectPtr<GemmSPNode> node = make_object<GemmSPNode>();
  auto a_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto e_access = NormalizeToAccessRegion(args[1], kAccessRead);
  auto b_access = NormalizeToAccessRegion(args[2], kAccessRead);
  auto c_access = NormalizeToAccessRegion(args[3], kAccessReadWrite);

  node->aRegion_ = a_access.region;
  node->eRegion_ = e_access.region;
  node->bRegion_ = b_access.region;
  node->cRegion_ = c_access.region;
  node->SetAccessRegions({a_access, e_access, b_access, c_access});

  node->A = RegionTensor(node->aRegion_);
  node->E = RegionTensor(node->eRegion_);
  node->B = RegionTensor(node->bRegion_);
  node->C = RegionTensor(node->cRegion_);

  node->trans_A = (GetIntegerArgument(args[4]) != 0);
  node->trans_E = (GetIntegerArgument(args[5]) != 0);
  node->trans_B = (GetIntegerArgument(args[6]) != 0);
  node->M = GetIntegerArgument(args[7]);
  node->N = GetIntegerArgument(args[8]);
  node->K = GetIntegerArgument(args[9]);
  node->policy = GemmSPWarpPolicy(GetIntegerArgument(args[10]));
  node->clear_accum = args[11].as<PrimExpr>().value();
  // wg_wait is a Hopper warpgroup knob set by the CUDA dialect; it rides in
  // the annotations rather than the positional call protocol.
  if (auto val = annotations.Get("wg_wait")) {
    node->wg_wait = static_cast<int>(GetIntegerAnnotation(*val));
  }
  if (auto val = annotations.Get("is_wgmma")) {
    node->isWgmma_ = GetBoolAnnotation(*val);
  }
  if (auto val = annotations.Get("is_tcgen05")) {
    node->isTcgen05_ = GetBoolAnnotation(*val);
  }

  data_ = std::move(node);
}

AccessRegions GemmSPNode::GetAccessRegions() const {
  AccessRegions result;
  result.reads.push_back(aRegion_);
  result.reads.push_back(eRegion_);
  result.reads.push_back(bRegion_);
  if (!IsOne(clear_accum)) {
    result.reads.push_back(cRegion_);
  }
  result.writes.push_back(cRegion_);
  return result;
}

ffi::Array<TensorRegion> GemmSPNode::GetReadBeforeWriteRegions() const {
  ffi::Array<TensorRegion> result;
  result.push_back(aRegion_);
  result.push_back(eRegion_);
  result.push_back(bRegion_);
  // See GemmNode::GetReadBeforeWriteRegions: only a provable absence of the
  // clear makes the accumulator's old contents a definite read.
  if (IsZero(clear_accum)) {
    result.push_back(cRegion_);
  }
  return result;
}

TileOperator GemmSPNode::Clone() const {
  auto op = make_object<GemmSPNode>(*this);
  return GemmSP(op);
}

String GemmSPNode::GetGemmSPInstructionKey(int block_size,
                                           Target target) const {
  return ResolveGemmSPImpl(target).select_inst(*this, block_size, target);
}

Stmt GemmSPNode::Lower(const LowerArgs &lower_args,
                       sym::AnalyzerObj *analyzer) const {
  const auto callback = ffi::Function::GetGlobal("tl.gemm_sp.lower");
  TVM_FFI_CHECK(callback.has_value(), ValueError)
      << "No lower function registered for tl.gemm_sp";
  return InlineTileOpFunction(
      (*callback)(GetRef<GemmSP>(this), lower_args.target,
                  lower_args.layout_map, lower_args.thread_bounds,
                  lower_args.thread_index),
      "tl.gemm_sp");
}

LayoutMap GemmSPNode::InferLayout(const LayoutInferArgs &layout_args,
                                  InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  if (const auto f = ffi::Function::GetGlobal("tl.gemm_sp.infer_layout")) {
    auto inferred_layouts = ((*f)(GetRef<GemmSP>(this), layout_args.target,
                                  layout_args.thread_bounds))
                                .template as_or_throw<LayoutMap>();
    auto block_size = TryConstInt(layout_args.thread_bounds->extent).value();
    String gemm_inst = GetGemmSPInstructionKey(block_size, layout_args.target);
    bool reuse_existing_shared_layout =
        ResolveGemmSPImpl(layout_args.target)
            .reuse_existing_shared_layout(gemm_inst);
    for (auto kv : inferred_layouts) {
      const TensorVar &buf = kv.first;
      const Layout &layout = kv.second;
      if (reuse_existing_shared_layout && IsSharedBuffer(buf) &&
          layout_args.layout_map.count(buf)) {
        continue;
      }
      if (auto frag = layout.as<Fragment>()) {
        results.Set(buf,
                    frag.value()->BindThreadRange(layout_args.thread_bounds));
      } else {
        results.Set(buf, layout);
      }
    }
  } else {
    LOG(FATAL) << "No infer layout function found for gemm_sp";
  }

  completed_ = true;
  return results;
}

TIR_REGISTER_TL_TILE_OP(GemmSP, gemm_sp)

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterTileOperator("tl.tileop.wgmma_gemm_sp",
                       [](Array<Expr> args, Map<String, Any> hints) {
                         hints.Set("is_wgmma", true);
                         return GemmSP(std::move(args), std::move(hints));
                       });
  RegisterTileOperator("tl.tileop.tcgen05_gemm_sp",
                       [](Array<Expr> args, Map<String, Any> hints) {
                         hints.Set("is_tcgen05", true);
                         return GemmSP(std::move(args), std::move(hints));
                       });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  GemmSPWarpPolicyNode::RegisterReflection();
  GemmSPNode::RegisterReflection();
  namespace refl = reflection;
  refl::GlobalDef().def("tl.GemmSPWarpPolicyComputeWarpPartition",
                        [](GemmSPWarpPolicy policy, int M, int N,
                           int block_size, Target target, String gemm_inst) {
                          policy->ComputeWarpPartition(M, N, block_size, target,
                                                       gemm_inst);
                        });
  refl::GlobalDef().def("tl.GemmSPGetGemmInstructionKey",
                        [](GemmSP gemm, int block_size, Target target) {
                          return gemm->GetGemmSPInstructionKey(block_size,
                                                               target);
                        });
}
} // namespace tl
} // namespace tvm
