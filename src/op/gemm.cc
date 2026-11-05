/*!
 * \file tl/op/gemm.cc
 * \brief Implementation of General Matrix Multiplication (GEMM) operators
 */

#include "gemm.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt.h>

#include "builtin.h"
#include "lowering.h"
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/math.h>
#include <tvm/tirx/op/memory.h>

#include "utils.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace prim;
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
          << target->str() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.gemm requires a target-specific implementation, but no gemm "
         "implementation is registered for "
      << target->str();
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
                                 const Map<String, Any> &annotations) {
  auto a_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto b_access = NormalizeToAccessRegion(args[1], kAccessRead);
  auto c_access = NormalizeToAccessRegion(args[2], kAccessReadWrite);

  node->aRegion_ = a_access.region;
  node->bRegion_ = b_access.region;
  node->cRegion_ = c_access.region;
  node->SetAccessRegions({a_access, b_access, c_access});

  node->a_ = RegionTensor(node->aRegion_);
  node->b_ = RegionTensor(node->bRegion_);
  node->c_ = RegionTensor(node->cRegion_);
  node->transA_ = (GetIntegerArgument(args[3]) != 0);
  node->transB_ = (GetIntegerArgument(args[4]) != 0);
  node->m_ = GetIntegerArgument(args[5]);
  node->n_ = GetIntegerArgument(args[6]);
  node->k_ = GetIntegerArgument(args[7]);
  node->policy_ = GemmWarpPolicy(GetIntegerArgument(args[8]));
  node->clearAccum_ = args[9].as<PrimExpr>().value();
  // k_pack rides in the annotations (a ROCm MFMA/WMMA lowering knob set by
  // the ROCm dialect), not in the positional call protocol.
  if (auto val = annotations.Get("k_pack")) {
    node->kPack_ = static_cast<int>(GetIntegerAnnotation(*val));
    ICHECK(node->kPack_ == 1 || node->kPack_ == 2) << "kPack must be 1 or 2";
  }
  // wg_wait is a Hopper warpgroup knob set by the CUDA dialect; like k_pack
  // it rides in the annotations rather than the positional call protocol.
  if (auto val = annotations.Get("wg_wait")) {
    node->wgWait_ = static_cast<int>(GetIntegerAnnotation(*val));
  }
  if (auto val = annotations.Get("is_wgmma")) {
    node->isWgmma_ = GetBoolAnnotation(*val);
  }
  if (auto val = annotations.Get("is_tcgen05")) {
    node->isTcgen05_ = GetBoolAnnotation(*val);
  }
  if (args.size() > 10 && args[10]->IsInstance<TensorLoadObj>()) {
    node->mbar_ = (args[10]).template as_or_throw<TensorLoad>();
  }
  node->cCoords_ = Array<PrimExpr>(
      {args[11].as<PrimExpr>().value(), args[12].as<PrimExpr>().value()});
  node->annotations_ = annotations;
}

Gemm::Gemm(Array<Expr> args, Map<String, Any> annotations) {
  ICHECK_EQ(args.size(), 13)
      << "tl.tileop.gemm expects exactly 13 positional slots, but got "
      << args.size()
      << "; a block-scaled GEMM (SFA, SFB, k_start) must be built as "
         "tl.tileop.gemm_blockscaled so the scale factors are not silently "
         "ignored.";
  ObjectPtr<GemmNode> node = make_object<GemmNode>();
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
                     sym::AnalyzerObj *analyzer) const {
  const auto callback = ffi::Function::GetGlobal("tl.gemm.lower");
  TVM_FFI_CHECK(callback.has_value(), ValueError)
      << "No lower function registered for tl.gemm";
  PrimExpr phase = lower_args.mbar_phase_expr;
  if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations_))
    phase = explicit_phase.value();
  return InlineTileOpFunction(
      (*callback)(GetRef<Gemm>(this), lower_args.layout_map, lower_args.target,
                  lower_args.thread_bounds, lower_args.thread_index, phase),
      "tl.gemm");
}

LayoutMap GemmNode::InferLayout(const LayoutInferArgs &layout_args,
                                InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  if (const auto f = ffi::Function::GetGlobal("tl.gemm.infer_layout")) {
    auto inferred_layouts = ((*f)(GetRef<Gemm>(this), layout_args.target,
                                  layout_args.thread_bounds))
                                .template as_or_throw<LayoutMap>();
    // For MMA instructions, skip shared buffer layouts that are already
    // inferred by a prior operator to avoid layout conflicts when the same
    // shared buffer is consumed by multiple gemm ops with different transpose
    // semantics. WGMMA/TCGEN5MMA have strict shared memory layout requirements
    // and must always set their layouts.
    auto block_size = TryConstInt(layout_args.thread_bounds->extent).value();
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

TIR_REGISTER_TL_TILE_OP(Gemm, gemm)

TVM_FFI_STATIC_INIT_BLOCK() {
  RegisterTileOperator("tl.tileop.wgmma_gemm",
                       [](Array<Expr> args, Map<String, Any> hints) {
                         hints.Set("is_wgmma", true);
                         return Gemm(std::move(args), std::move(hints));
                       });
  RegisterTileOperator("tl.tileop.tcgen05_gemm",
                       [](Array<Expr> args, Map<String, Any> hints) {
                         hints.Set("is_tcgen05", true);
                         return Gemm(std::move(args), std::move(hints));
                       });
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
