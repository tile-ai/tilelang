/*!
 * \file tl/op/gemm.cc
 * \brief Implementation of General Matrix Multiplication (GEMM) operators
 */

#include "gemm.h"
#include "support/check.h"
#include <tvm/ffi/error.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/stmt.h>

#include "builtin.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/region.h>

#include "utils.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

std::vector<GemmImpl> &GemmImplRegistry() {
  static std::vector<GemmImpl> registry;
  return registry;
}

} // namespace

const GemmImpl &ResolveGemmImpl(const Target &target) {
  const auto &registry = GemmImplRegistry();
  const GemmImpl *matched_impl = nullptr;
  for (const GemmImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.gemm found multiple target-specific implementations for "
          << target << ": " << matched_impl->name << " and " << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.gemm requires a target-specific implementation, but no gemm "
         "implementation is registered for "
      << target;
  return *matched_impl;
}

void RegisterGemmImpl(GemmImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.select_inst != nullptr);
  ICHECK(impl.compute_warp_partition != nullptr);
  ICHECK(impl.reuse_existing_shared_layout != nullptr);
  GemmImplRegistry().push_back(impl);
}

/**
 * @brief Construct a Gemm operator from serialized TL arguments.
 *
 * Deserializes operator parameters from `args` and resolves buffer references,
 * populating an internal GemmNode with buffers, transpose flags, M/N/K,
 * warp policy, clear_accum, an optional mbarrier operand and the C tile
 * coordinates.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Bptr, Cptr, trans_A (Bool), trans_B (Bool),
 *      M (Int), N (Int), K (Int), policy (Int), clear_accum (Bool),
 *      (optional) mbar (TensorLoad or const-0 placeholder),
 *      cCoord_y (PrimExpr), cCoord_x (PrimExpr)]
 *   Block-scaled GEMM appends [SFA, SFB regions, k_start (PrimExpr)] and is
 *   built through tl.tileop.gemm_blockscaled (GemmBlockScaled) instead.
 *   Backend lowering knobs (k_pack, wg_wait) ride in the annotations map.
 */
void GemmNode::InitFromDenseArgs(GemmNode *node, const Array<Expr> &args,
                                 const BlockAnnotations &annotations) {
  auto a_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto b_access = NormalizeToAccessRegion(args[1], kAccessRead);
  auto c_access = NormalizeToAccessRegion(args[2], kAccessReadWrite);

  node->aRegion_ = a_access.region;
  node->bRegion_ = b_access.region;
  node->cRegion_ = c_access.region;
  node->SetAccessRegions({a_access, b_access, c_access});

  node->a_ = node->aRegion_->source.as_or_throw<TensorVar>();
  node->b_ = node->bRegion_->source.as_or_throw<TensorVar>();
  node->c_ = node->cRegion_->source.as_or_throw<TensorVar>();
  node->transA_ = GetConstInt(args[3]).value() != 0;
  node->transB_ = GetConstInt(args[4]).value() != 0;
  node->m_ = GetConstInt(args[5]).value();
  node->n_ = GetConstInt(args[6]).value();
  node->k_ = GetConstInt(args[7]).value();
  node->policy_ = GemmWarpPolicy(GetConstInt(args[8]).value());
  node->clearAccum_ = args[9].as<PrimExpr>().value();
  // k_pack rides in the annotations (a ROCm MFMA/WMMA lowering knob set by
  // the ROCm dialect), not in the positional call protocol.
  if (auto val = annotations.Get("k_pack")) {
    auto value = GetIntAnnotation(val.value());
    TVM_FFI_CHECK(value.has_value(), ValueError)
        << "k_pack annotation must be IntImmNode";
    node->kPack_ = value.value();
    ICHECK(node->kPack_ == 1 || node->kPack_ == 2) << "kPack must be 1 or 2";
  }
  // wg_wait is a Hopper warpgroup knob set by the CUDA dialect; like k_pack
  // it rides in the annotations rather than the positional call protocol.
  if (auto val = annotations.Get("wg_wait")) {
    auto value = GetIntAnnotation(val.value());
    TVM_FFI_CHECK(value.has_value(), ValueError)
        << "wg_wait annotation must be IntImmNode";
    node->wgWait_ = value.value();
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
  if (args.size() > 10 && args[10]->IsInstance<TensorLoadNode>()) {
    node->mbar_ = args[10].as_or_throw<TensorLoad>();
  } else if (args.size() > 10 && args[10]->IsInstance<TensorRegionNode>()) {
    auto region = args[10].as_or_throw<TensorRegion>();
    Array<PrimExpr> indices;
    for (const Range &axis : region->region) {
      TVM_FFI_CHECK(IsOne(axis->extent), ValueError)
          << "GEMM requires a single mbarrier element";
      indices.push_back(axis->begin);
    }
    node->mbar_ = TensorLoad(region->source, indices, region->loc);
  } else if (args.size() > 10) {
    TVM_FFI_CHECK(GetConstInt(args[10]).value_or(-1) == 0, ValueError)
        << "GEMM mbarrier must be a tensor element, a single-element region, "
           "or zero";
  }
  node->cCoords_ = Array<PrimExpr>(
      {args[11].as<PrimExpr>().value(), args[12].as<PrimExpr>().value()});
  node->annotations_ = annotations;
}

Gemm::Gemm(Array<Expr> args, BlockAnnotations annotations) {
  ICHECK_EQ(args.size(), 13)
      << "tl.tileop.gemm expects exactly 13 positional slots, but got "
      << args.size()
      << "; a block-scaled GEMM (SFA, SFB, k_start) must be built as "
         "tl.tileop.gemm_blockscaled so the scale factors are not silently "
         "ignored.";
  ObjectPtr<GemmNode> node = make_object<GemmNode>(ffi::UnsafeInit{});
  GemmNode::InitFromDenseArgs(node.get(), args, annotations);
  data_ = std::move(node);
}

AccessRegions GemmNode::GetAccessRegions() const {
  AccessRegions result;
  result.reads.push_back(aRegion_);
  result.reads.push_back(bRegion_);
  if (!IsOne(clearAccum_)) {
    result.reads.push_back(cRegion_);
  }
  result.writes.push_back(cRegion_);
  return result;
}

ffi::Array<TensorRegion> GemmNode::GetReadBeforeWriteRegions() const {
  ffi::Array<TensorRegion> result;
  result.push_back(aRegion_);
  result.push_back(bRegion_);
  // The accumulator's old contents are consumed only when the clear is
  // provably absent. GetAccessRegions() uses !IsOne() because a clear that
  // cannot be proven still creates a read dependency for pipelining; here the
  // question is whether the op definitely reads, so the pipelined idiom
  // `clear_accum=(k == 0)` must not count.
  if (IsZero(clearAccum_)) {
    result.push_back(cRegion_);
  }
  return result;
}

TileOperator GemmNode::Clone() const {
  auto op = make_object<GemmNode>(*this);
  return Gemm(op);
}

String GemmNode::GetGemmInstructionKey(int block_size, Target target) const {
  return ResolveGemmImpl(target).select_inst(*this, block_size, target);
}

std::pair<int, int> GemmWarpPolicyNode::ComputeWarpPartition(
    int M, int N, int block_size, Target target, String gemm_inst) const {
  return ResolveGemmImpl(target).compute_warp_partition(*this, M, N, block_size,
                                                        target, gemm_inst);
}

Stmt GemmNode::Lower(const LowerArgs &lower_args,
                     sym::Analyzer analyzer) const {
  GetGemmInstructionKey(GetConstInt(lower_args.thread_bounds->extent).value(),
                        lower_args.target);
  const auto &implementation = ResolveGemmImpl(lower_args.target);
  if (implementation.lower) {
    return implementation.lower(*this, lower_args, analyzer);
  }
  auto lower = ffi::Function::GetGlobal("tl.gemm.lower");
  TVM_FFI_CHECK(lower.has_value(), ValueError)
      << "No lower function found for gemm";
  PrimExpr mbar_phase = lower_args.mbar_phase_expr;
  if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations_)) {
    mbar_phase = explicit_phase.value();
  }
  auto function =
      (*lower)(GetRef<Gemm>(this), lower_args.layout_map, lower_args.target,
               lower_args.thread_bounds, lower_args.thread_index, mbar_phase)
          .cast<tirx::Function>();
  auto name = function->GetAttr<String>("global_symbol");
  TVM_FFI_CHECK(name.has_value(), ValueError)
      << "GEMM lowering must return a named native function";
  return RegionStmt(Op::Get("tirx.scope"), {}, {},
                    DictAttrs({{"name_hint", name.value()},
                               {tl::attr::kLexicalAllocScope, true}}),
                    function->body.value());
}

LayoutMap GemmNode::InferLayout(const LayoutInferArgs &layout_args,
                                InferLevel level) const {
  if (completed_)
    return {};
  GetGemmInstructionKey(GetConstInt(layout_args.thread_bounds->extent).value(),
                        layout_args.target);
  const auto &implementation = ResolveGemmImpl(layout_args.target);
  if (implementation.infer_layout) {
    completed_ = true;
    return implementation.infer_layout(*this, layout_args);
  }
  LayoutMap results;
  if (const auto f = ffi::Function::GetGlobal("tl.gemm.infer_layout")) {
    auto inferred_layouts =
        (*f)(GetRef<Gemm>(this), layout_args.target, layout_args.thread_bounds)
            .cast<LayoutMap>();
    // For MMA instructions, skip shared buffer layouts that are already
    // inferred by a prior operator to avoid layout conflicts when the same
    // shared buffer is consumed by multiple gemm ops with different transpose
    // semantics. WGMMA/TCGEN5MMA have strict shared memory layout requirements
    // and must always set their layouts.
    auto block_size = GetConstInt(layout_args.thread_bounds->extent).value();
    String gemm_inst = GetGemmInstructionKey(block_size, layout_args.target);
    bool reuse_existing_shared_layout =
        ResolveGemmImpl(layout_args.target)
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
    LOG(FATAL) << "No infer layout function found for gemm";
  }

  completed_ = true;
  return results;
}

TIR_REGISTER_TL_TILE_OP(Gemm, gemm, -1)

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const auto &[name, annotation] : std::vector<std::pair<String, String>>{
           {"tl.tileop.wgmma_gemm", "is_wgmma"},
           {"tl.tileop.tcgen05_gemm", "is_tcgen05"}}) {
    RegisterBuiltin<-1>(name, CallEffectKind::kOpaque);
    OpDef(name)
        .set_attr(tvm::op_attr::kFixedReturnType, TupleType(Array<Type>{}))
        .set_attr<OpBuilderFunc>(
            "TLOpBuilder",
            [annotation](Array<Expr> args, BlockAnnotations annotations) {
              annotations.Set(annotation, true);
              return Gemm(std::move(args), std::move(annotations));
            });
  }
}

TVM_FFI_STATIC_INIT_BLOCK() {
  GemmNode::RegisterReflection();
  GemmWarpPolicyNode::RegisterReflection();
  namespace refl = reflection;
  refl::GlobalDef().def("tl.GemmWarpPolicyComputeWarpPartition",
                        [](GemmWarpPolicy policy, int M, int N, int block_size,
                           Target target, String gemm_inst) {
                          policy->ComputeWarpPartition(M, N, block_size, target,
                                                       gemm_inst);
                        });
  refl::GlobalDef().def("tl.GemmGetGemmInstructionKey",
                        [](Gemm gemm, int block_size, Target target) {
                          return gemm->GetGemmInstructionKey(block_size,
                                                             target);
                        });
}

} // namespace tl
} // namespace tvm
