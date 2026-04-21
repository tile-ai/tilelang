/*!
 * \file tl/op/gemm_sp.cc
 *
 * Define gemm_sp operator.
 */

#include "gemm_sp.h"
#include "utils.h"

#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

#include "../target/utils.h"
#include "tvm/ffi/string.h"

namespace tvm {
namespace tl {

using namespace tir;

std::pair<int, int>
GemmSPWarpPolicyNode::computeWarpPartition(int M, int N, int block_size,
                                           Target target, GemmInst gemm_inst,
                                           int bits) const {
  int num_warps = block_size / TargetGetWarpSize(target);

  ICHECK(gemm_inst == GemmInst::kMMA || gemm_inst == GemmInst::kWGMMA)
      << "GemmSP currently only supports MMA and WGMMA";
  auto [m_warp, n_warp] = GemmWarpPolicyNode::computeWarpPartition(
      M, N, block_size, target, gemm_inst);

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
 * @brief Construct a GemmSP operator from serialized TL arguments.
 *
 * Deserializes operator parameters from `args` and resolves buffer references,
 * populating an internal GemmSPNode with buffers, transpose flags, M/N/K,
 * warp policy, clear_accum, strides, offsets, and optional kPack/wg_wait.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Eptr, Bptr, Cptr, trans_A (Bool), trans_B (Bool),
 *      trans_E (Bool), M (Int), N (Int), K (Int), policy (Int),
 *      clear_accum (Bool), stride_A (Int), stride_B (Int),
 *      offset_A (Int), offset_B (Int),
 *      (optional) kPack (Int), (optional) wg_wait (Int)]
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

  node->A = node->aRegion_->buffer;
  node->E = node->eRegion_->buffer;
  node->B = node->bRegion_->buffer;
  node->C = node->cRegion_->buffer;

  node->trans_A = args[4].as<Bool>().value();
  node->trans_B = args[5].as<Bool>().value();
  node->trans_E = args[6].as<Bool>().value();
  node->M = args[7].as<IntImm>().value()->value;
  node->N = args[8].as<IntImm>().value()->value;
  node->K = args[9].as<IntImm>().value()->value;
  node->policy = GemmWarpPolicy(args[10].as<IntImm>().value()->value);
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

/**
 * @brief Create a deep copy of this GemmSPNode wrapped as a TileOperator.
 *
 * @return TileOperator A TileOperator holding a cloned GemmSPNode.
 */
TileOperator GemmSPNode::Clone() const {
  auto op = tvm::ffi::make_object<GemmSPNode>(*this);
  return GemmSP(op);
}

GemmInst GemmSPNode::GetGemmInst(int block_size, Target target) const {
  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  bool allow_wgmma = TargetIsHopper(target) && (this->M >= 64) &&
                     (num_warps % 4 == 0) && CheckWGMMA();
  if (allow_wgmma) {
    return GemmInst::kWGMMA;
  } else if (TargetIsCDNA(target)) {
    return GemmInst::kMFMA;
  } else if (TargetIsRDNA(target)) {
    return GemmInst::kWMMA;
  } else if (TargetIsCuda(target)) {
    return GemmInst::kMMA;
  } else {
    ICHECK(0) << "Unsupported target for gemm: " << target->str();
  }
}

bool GemmSPNode::CheckWGMMA() const {
  return false; // not supported yet
}

Stmt GemmSPNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (const auto f = ffi::Function::GetGlobal("tl.gemm_sp.lower")) {
    auto prim_func =
        Downcast<PrimFunc>((*f)(tvm::ffi::GetRef<GemmSP>(this), T.target,
                                T.thread_bounds, T.thread_var));
    ICHECK(prim_func->attrs.defined());
    auto global_symbol = prim_func->attrs.GetAttr<String>("global_symbol");
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
    LOG(FATAL) << "No lower function found for gemm_sp";
  }
}

LayoutMap GemmSPNode::InferLayout(const LayoutInferArgs &T,
                                  InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;

  if (const auto f = ffi::Function::GetGlobal("tl.gemm_sp.infer_layout")) {
    results = Downcast<LayoutMap>(
        (*f)(tvm::ffi::GetRef<GemmSP>(this), T.target, T.thread_bounds));
  } else {
    LOG(FATAL) << "No infer layout function found for gemm_sp";
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
         GemmInst gemm_inst, int bits) {
        policy->computeWarpPartition(M, N, block_size, target, gemm_inst, bits);
        return;
      });
}
} // namespace tl
} // namespace tvm
