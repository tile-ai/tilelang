/*!
 * \file tl/op/gemm.cc
 * \brief Implementation of General Matrix Multiplication (GEMM) operators
 */

#include "gemm.h"

#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

#include "utils.h"

#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

std::vector<GemmImpl> &GemmImplRegistry() {
  static std::vector<GemmImpl> registry;
  return registry;
}

const GemmImpl &ResolveGemmImpl(Target target) {
  const auto &registry = GemmImplRegistry();
  const GemmImpl *matched_impl = nullptr;
  for (const GemmImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.gemm found multiple target-specific implementations for "
          << target->ToDebugString() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.gemm requires a target-specific implementation, but no gemm "
         "implementation is registered for "
      << target->ToDebugString();
  return *matched_impl;
}

} // namespace

void RegisterGemmImpl(GemmImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.select_inst != nullptr);
  ICHECK(impl.compute_warp_partition != nullptr);
  ICHECK(impl.reuse_existing_shared_layout != nullptr);
  ICHECK(impl.instruction_kind != nullptr);
  GemmImplRegistry().push_back(impl);
}

/**
 * @brief Construct a Gemm operator from serialized TL arguments.
 *
 * Deserializes operator parameters from `args` and resolves buffer references,
 * populating an internal GemmNode with buffers, transpose flags, M/N/K,
 * warp policy, clear_accum, strides, offsets, optional kPack/wg_wait, and
 * optional mbarrier.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Bptr, Cptr, trans_A (Bool), trans_B (Bool),
 *      M (Int), N (Int), K (Int), policy (Int), clear_accum (Bool),
 *      stride_A (Int), stride_B (Int), offset_A (Int), offset_B (Int),
 *      (optional) kPack (Int), (optional) internal wg_wait (Int),
 *      (optional) mbar (BufferLoad), cCoord_y (PrimExpr), cCoord_x (PrimExpr)]
 */
Gemm::Gemm(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<GemmNode> node = tvm::ffi::make_object<GemmNode>();

  auto a_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto b_access = NormalizeToAccessRegion(args[1], kAccessRead);
  auto c_access = NormalizeToAccessRegion(args[2], kAccessReadWrite);

  node->aRegion_ = a_access.region;
  node->bRegion_ = b_access.region;
  node->cRegion_ = c_access.region;
  node->SetAccessRegions({a_access, b_access, c_access});

  node->a_ = node->aRegion_->buffer;
  node->b_ = node->bRegion_->buffer;
  node->c_ = node->cRegion_->buffer;
  node->transA_ = args[3].as<Bool>().value();
  node->transB_ = args[4].as<Bool>().value();
  node->m_ = args[5].as<IntImm>().value()->value;
  node->n_ = args[6].as<IntImm>().value()->value;
  node->k_ = args[7].as<IntImm>().value()->value;
  node->policy_ = GemmWarpPolicy(args[8].as<IntImm>().value()->value);
  node->clearAccum_ = args[9].as<PrimExpr>().value();
  node->strideA_ = args[10].as<IntImm>().value()->value;
  node->strideB_ = args[11].as<IntImm>().value()->value;
  node->offsetA_ = args[12].as<IntImm>().value()->value;
  node->offsetB_ = args[13].as<IntImm>().value()->value;
  if (args.size() > 14) {
    node->kPack_ = args[14].as<IntImm>().value()->value;
    if (node->kPack_ != 1 && node->kPack_ != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 15) {
    node->wgWait_ = args[15].as<IntImm>().value()->value;
  }
  if (auto val = annotations.Get("is_wgmma")) {
    const auto *int_val = val->as<IntImmNode>();
    ICHECK(int_val) << "is_wgmma annotation must be IntImmNode";
    node->isWgmma_ = int_val->value != 0;
  }
  if (auto val = annotations.Get("is_tcgen05")) {
    const auto *int_val = val->as<IntImmNode>();
    ICHECK(int_val) << "is_tcgen05 annotation must be IntImmNode";
    node->isTcgen05_ = int_val->value != 0;
  }
  if (args.size() > 16 && args[16]->IsInstance<BufferLoadNode>()) {
    node->mbar_ = Downcast<BufferLoad>(args[16]);
  }
  node->cCoords_ = Array<PrimExpr>(
      {args[17].as<PrimExpr>().value(), args[18].as<PrimExpr>().value()});
  if (args.size() > 19) {
    node->sfaRegion_ = NormalizeToBufferRegion(args[19]);
  }
  if (args.size() > 20) {
    node->sfbRegion_ = NormalizeToBufferRegion(args[20]);
  }
  if (args.size() > 21) {
    node->sfAId_ = args[21].as<PrimExpr>().value();
  }
  if (args.size() > 22) {
    node->sfBId_ = args[22].as<PrimExpr>().value();
  }
  node->annotations_ = annotations;
  data_ = std::move(node);
}

AccessRegions GemmNode::GetAccessRegions() const {
  AccessRegions result;
  result.reads.push_back(aRegion_);
  result.reads.push_back(bRegion_);
  if (!is_one(clearAccum_)) {
    result.reads.push_back(cRegion_);
  }
  if (sfaRegion_.defined()) {
    result.reads.push_back(sfaRegion_);
  }
  if (sfbRegion_.defined()) {
    result.reads.push_back(sfbRegion_);
  }
  result.writes.push_back(cRegion_);
  return result;
}

TileOperator GemmNode::Clone() const {
  auto op = tvm::ffi::make_object<GemmNode>(*this);
  return Gemm(op);
}

String GemmNode::getGemmInstructionKey(int block_size, Target target) const {
  return ResolveGemmImpl(target).select_inst(*this, block_size, target);
}

String GemmNode::getGemmInstructionKind(int block_size, Target target) const {
  const GemmImpl &impl = ResolveGemmImpl(target);
  return impl.instruction_kind(impl.select_inst(*this, block_size, target));
}

std::pair<int, int> GemmWarpPolicyNode::computeWarpPartition(
    int M, int N, int block_size, Target target, String gemm_inst) const {
  return ResolveGemmImpl(target).compute_warp_partition(*this, M, N, block_size,
                                                        target, gemm_inst);
}

Stmt GemmNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (const auto f = ffi::Function::GetGlobal("tl.gemm.lower")) {
    PrimExpr mbar_phase = T.mbar_phase_expr;
    if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations_)) {
      mbar_phase = explicit_phase.value();
    }
    // NOTE(wt): Decide the instruction key and compute warp partition on Python
    // side.
    auto prim_func = Downcast<PrimFunc>(
        (*f)(tvm::ffi::GetRef<Gemm>(this), T.layout_map, T.target,
             T.thread_bounds, T.thread_var, mbar_phase));
    ICHECK(prim_func->attrs.defined());
    auto global_symbol =
        prim_func->attrs.GetAttr<tvm::ffi::String>("global_symbol");
    ICHECK(global_symbol.has_value());
    if (prim_func->body.as<BlockRealizeNode>()) {
      BlockRealize block_realize = Downcast<BlockRealize>(prim_func->body);
      auto block = block_realize->block;
      {
        BlockNode *n = block.CopyOnWrite();
        n->name_hint = global_symbol.value();
        n->annotations.Set(tl::attr::kLexicalAllocScope,
                           IntImm(DataType::Int(32), 1));
      }
      return BlockRealize(block_realize->iter_values, block_realize->predicate,
                          block);
    }
    // wrap with block realize node
    Map<String, ObjectRef> block_annotations;
    block_annotations.Set(tl::attr::kLexicalAllocScope,
                          IntImm(DataType::Int(32), 1));
    return BlockRealize(
        /*iter_values=*/Array<PrimExpr>(),
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
              /*name_hint=*/global_symbol.value(), prim_func->body,
              /*init=*/Optional<Stmt>(), /*alloc_buffers=*/{},
              /*match_buffers=*/{}, /*annotations=*/block_annotations));
  } else {
    LOG(FATAL) << "No lower function found for gemm";
    return Stmt();
  }
}

LayoutMap GemmNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  if (const auto f = ffi::Function::GetGlobal("tl.gemm.infer_layout")) {
    auto inferred_layouts = Downcast<LayoutMap>(
        (*f)(tvm::ffi::GetRef<Gemm>(this), T.target, T.thread_bounds));
    // For MMA instructions, skip shared buffer layouts that are already
    // inferred by a prior operator to avoid layout conflicts when the same
    // shared buffer is consumed by multiple gemm ops with different transpose
    // semantics. WGMMA/TCGEN5MMA have strict shared memory layout requirements
    // and must always set their layouts.
    auto block_size = *as_const_int(T.thread_bounds->extent);
    String gemm_inst = getGemmInstructionKey(block_size, T.target);
    bool reuse_existing_shared_layout =
        ResolveGemmImpl(T.target).reuse_existing_shared_layout(gemm_inst);
    for (auto kv : inferred_layouts) {
      const Buffer &buf = kv.first;
      const Layout &layout = kv.second;
      if (reuse_existing_shared_layout && IsSharedBuffer(buf) &&
          T.layout_map.count(buf)) {
        continue;
      }
      if (auto frag = layout.as<Fragment>()) {
        results.Set(buf, frag.value()->BindThreadRange(T.thread_bounds));
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
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.wgmma_gemm")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "wgmma_gemm")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_wgmma",
                                       IntImm(DataType::Int(32), 1));
                               return Gemm(args, ann);
                             })
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.tcgen05_gemm")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "tcgen05_gemm")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_tcgen05",
                                       IntImm(DataType::Int(32), 1));
                               return Gemm(args, ann);
                             })
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.GemmWarpPolicy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "GemmWarpPolicy");

TVM_FFI_STATIC_INIT_BLOCK() {
  GemmNode::RegisterReflection();
  GemmWarpPolicyNode::RegisterReflection();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.GemmWarpPolicyComputeWarpPartition",
                        [](GemmWarpPolicy policy, int M, int N, int block_size,
                           Target target, String gemm_inst) {
                          policy->computeWarpPartition(M, N, block_size, target,
                                                       gemm_inst);
                        });
  refl::GlobalDef().def("tl.GemmGetGemmInstructionKey",
                        [](Gemm gemm, int block_size, Target target) {
                          return gemm->getGemmInstructionKey(block_size,
                                                             target);
                        });
}

} // namespace tl
} // namespace tvm
