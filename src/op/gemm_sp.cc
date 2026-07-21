/*!
 * \file tl/op/gemm_sp.cc
 *
 * Define gemm_sp operator.
 */

#include "gemm_sp.h"
#include "support/check.h"
#include "utils.h"

#include "builtin.h"
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
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
 * warp policy, clear_accum, strides, offsets, and optional kPack/wg_wait.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Eptr, Bptr, Cptr, trans_A (Bool), trans_E (Bool),
 *      trans_B (Bool), M (Int), N (Int), K (Int), policy (Int),
 *      clear_accum (Bool), stride_A (Int), stride_B (Int),
 *      offset_A (Int), offset_B (Int),
 *      (optional) kPack (Int), (optional) wg_wait (Int)]
 */
GemmSP::GemmSP(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
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

  node->A = node->aRegion_->buffer;
  node->E = node->eRegion_->buffer;
  node->B = node->bRegion_->buffer;
  node->C = node->cRegion_->buffer;

  node->trans_A = args[4].as<Bool>().value();
  node->trans_E = args[5].as<Bool>().value();
  node->trans_B = args[6].as<Bool>().value();
  node->M = args[7].as<IntImm>().value()->value;
  node->N = args[8].as<IntImm>().value()->value;
  node->K = args[9].as<IntImm>().value()->value;
  node->policy = GemmSPWarpPolicy(args[10].as<IntImm>().value()->value);
  node->clear_accum = args[11].as<PrimExpr>().value();
  node->stride_A = args[12].as<IntImm>().value()->value;
  node->stride_B = args[13].as<IntImm>().value()->value;
  node->offset_A = args[14].as<IntImm>().value()->value;
  node->offset_B = args[15].as<IntImm>().value()->value;
  if (args.size() > 16) {
    node->kPack = args[16].as<IntImm>().value()->value;
    if (node->kPack != 1 && node->kPack != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 17) {
    node->wg_wait = args[17].as<IntImm>().value()->value;
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

  data_ = std::move(node);
}

AccessRegions GemmSPNode::GetAccessRegions() const {
  AccessRegions result;
  result.reads.push_back(aRegion_);
  result.reads.push_back(eRegion_);
  result.reads.push_back(bRegion_);
  if (!is_one(clear_accum)) {
    result.reads.push_back(cRegion_);
  }
  result.writes.push_back(cRegion_);
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
                       arith::Analyzer *analyzer) const {
  if (const auto f = Function::GetGlobal("tl.gemm_sp.lower")) {
    auto prim_func = Downcast<PrimFunc>(
        (*f)(GetRef<GemmSP>(this), lower_args.target, lower_args.layout_map,
             lower_args.thread_bounds, lower_args.thread_index));
    ICHECK(prim_func->attrs.defined());
    auto global_symbol = prim_func->attrs.GetAttr<String>("global_symbol");
    ICHECK(global_symbol.has_value());
    if (prim_func->body.as<SBlockRealizeNode>()) {
      SBlockRealize block_realize = Downcast<SBlockRealize>(prim_func->body);
      auto block = block_realize->block;
      {
        SBlockNode *n = block.CopyOnWrite();
        n->name_hint = global_symbol.value();
        n->annotations.Set(tl::attr::kLexicalAllocScope,
                           IntImm(DataType::Int(32), 1));
      }
      return SBlockRealize(block_realize->iter_values, block_realize->predicate,
                           block);
    }
    // wrap with block realize node
    Map<String, ObjectRef> block_annotations;
    block_annotations.Set(tl::attr::kLexicalAllocScope,
                          IntImm(DataType::Int(32), 1));
    return SBlockRealize(
        /*iter_values=*/Array<PrimExpr>(),
        /*predicate=*/const_true(),
        /*block=*/
        SBlock(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
               /*name_hint=*/global_symbol.value(), prim_func->body,
               /*init=*/Optional<Stmt>(), /*alloc_buffers=*/{},
               /*match_buffers=*/{}, /*annotations=*/block_annotations));
  } else {
    LOG(FATAL) << "No lower function found for gemm_sp";
    return Stmt();
  }
}

LayoutMap GemmSPNode::InferLayout(const LayoutInferArgs &layout_args,
                                  InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  if (const auto f = Function::GetGlobal("tl.gemm_sp.infer_layout")) {
    auto inferred_layouts = Downcast<LayoutMap>((*f)(
        GetRef<GemmSP>(this), layout_args.target, layout_args.thread_bounds));
    auto block_size = *as_const_int(layout_args.thread_bounds->extent);
    String gemm_inst = GetGemmSPInstructionKey(block_size, layout_args.target);
    bool reuse_existing_shared_layout =
        ResolveGemmSPImpl(layout_args.target)
            .reuse_existing_shared_layout(gemm_inst);
    for (auto kv : inferred_layouts) {
      const Buffer &buf = kv.first;
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
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.wgmma_gemm_sp")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "wgmma_gemm_sp")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_wgmma",
                                       IntImm(DataType::Int(32), 1));
                               return GemmSP(args, ann);
                             })
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.tcgen05_gemm_sp")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "tcgen05_gemm_sp")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_tcgen05",
                                       IntImm(DataType::Int(32), 1));
                               return GemmSP(args, ann);
                             })
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

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
