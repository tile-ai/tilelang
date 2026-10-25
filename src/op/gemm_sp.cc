/*!
 * \file tl/op/gemm_sp.cc
 *
 * Define gemm_sp operator.
 */

#include "gemm_sp.h"
#include "support/check.h"
#include "utils.h"

#include "builtin.h"
#include <tvm/ffi/error.h>
#include <tvm/ir/analysis.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt.h>

#include "backend/common/target_utils.h"
#include "tvm/ffi/string.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
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
          << target << ": " << matched_impl->name << " and " << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.gemm_sp requires a target-specific implementation, but no "
         "gemm_sp implementation is registered for "
      << target;
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
GemmSP::GemmSP(Array<Expr> args, BlockAnnotations annotations) {
  TVM_FFI_CHECK_EQ(args.size(), 12, ValueError)
      << "gemm_sp expects 12 arguments";
  ObjectPtr<GemmSPNode> node = make_object<GemmSPNode>(ffi::UnsafeInit{});
  auto a_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto e_access = NormalizeToAccessRegion(args[1], kAccessRead);
  auto b_access = NormalizeToAccessRegion(args[2], kAccessRead);
  auto c_access = NormalizeToAccessRegion(args[3], kAccessReadWrite);

  node->aRegion_ = a_access.region;
  node->eRegion_ = e_access.region;
  node->bRegion_ = b_access.region;
  node->cRegion_ = c_access.region;
  node->SetAccessRegions({a_access, e_access, b_access, c_access});

  node->A = node->aRegion_->source.as_or_throw<TensorVar>();
  node->E = node->eRegion_->source.as_or_throw<TensorVar>();
  node->B = node->bRegion_->source.as_or_throw<TensorVar>();
  node->C = node->cRegion_->source.as_or_throw<TensorVar>();

  node->trans_A = GetConstInt(args[4]).value() != 0;
  node->trans_E = GetConstInt(args[5]).value() != 0;
  node->trans_B = GetConstInt(args[6]).value() != 0;
  node->M = GetConstInt(args[7]).value();
  node->N = GetConstInt(args[8]).value();
  node->K = GetConstInt(args[9]).value();
  node->policy = GemmSPWarpPolicy(GetConstInt(args[10]).value());
  node->clear_accum = args[11].as<PrimExpr>().value();
  // wg_wait is a Hopper warpgroup knob set by the CUDA dialect; it rides in
  // the annotations rather than the positional call protocol.
  if (auto val = annotations.Get("wg_wait")) {
    auto value = GetIntAnnotation(val.value());
    TVM_FFI_CHECK(value.has_value(), ValueError)
        << "wg_wait annotation must be IntImmNode";
    node->wg_wait = value.value();
  }
  if (auto val = annotations.Get("is_wgmma")) {
    auto value = GetIntAnnotation(val.value());
    TVM_FFI_CHECK(value.has_value(), ValueError)
        << "is_wgmma annotation must be IntImmNode";
    node->isWgmma_ = value.value() != 0;
  }
  if (auto val = annotations.Get("is_tcgen05")) {
    auto value = GetIntAnnotation(val.value());
    TVM_FFI_CHECK(value.has_value(), ValueError)
        << "is_tcgen05 annotation must be IntImmNode";
    node->isTcgen05_ = value.value() != 0;
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
                       sym::Analyzer analyzer) const {
  auto lower = ffi::Function::GetGlobal("tl.gemm_sp.lower");
  TVM_FFI_CHECK(lower.has_value(), ValueError)
      << "No lower function found for gemm_sp";
  auto function =
      (*lower)(GetRef<GemmSP>(this), lower_args.layout_map, lower_args.target,
               lower_args.thread_bounds, lower_args.thread_index)
          .cast<tirx::Function>();
  auto name = function->GetAttr<String>("global_symbol");
  TVM_FFI_CHECK(name.has_value(), ValueError)
      << "GEMM lowering must return a named native function";
  return RegionStmt(Op::Get("tirx.scope"), {}, {},
                    DictAttrs({{"name_hint", name.value()},
                               {tl::attr::kLexicalAllocScope, true}}),
                    function->body.value());
}

LayoutMap GemmSPNode::InferLayout(const LayoutInferArgs &layout_args,
                                  InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  if (const auto f = ffi::Function::GetGlobal("tl.gemm_sp.infer_layout")) {
    auto inferred_layouts = (*f)(GetRef<GemmSP>(this), layout_args.target,
                                 layout_args.thread_bounds)
                                .cast<LayoutMap>();
    auto block_size = GetConstInt(layout_args.thread_bounds->extent).value();
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

TIR_REGISTER_TL_TILE_OP(GemmSP, gemm_sp, -1)

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const auto &[name, annotation] : std::vector<std::pair<String, String>>{
           {"tl.tileop.wgmma_gemm_sp", "is_wgmma"},
           {"tl.tileop.tcgen05_gemm_sp", "is_tcgen05"}}) {
    RegisterBuiltin<-1>(name, CallEffectKind::kOpaque);
    OpDef(name)
        .set_attr(tvm::op_attr::kFixedReturnType, TupleType(Array<Type>{}))
        .set_attr<OpBuilderFunc>(
            "TLOpBuilder",
            [annotation](Array<Expr> args, BlockAnnotations annotations) {
              annotations.Set(annotation, true);
              return GemmSP(std::move(args), std::move(annotations));
            });
  }
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
