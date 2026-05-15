/*!
 * \file tl/op/gemm_sp_py.cc
 * \brief Implementation of Sparse General Matrix Multiplication (GEMM_SP)
 * operators
 */

#include "gemm_sp_py.h"
#include "utils.h"

#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

#include "tvm/ffi/string.h"

namespace tvm {
namespace tl {

using namespace tir;

/**
 * @brief Construct a Gemm operator from serialized TL arguments and a buffer
 * map.
 *
 * This constructor deserializes operator parameters from `args` and resolves
 * buffer references via `vmap`, populating an internal GemmSPPyNode with:
 * - device pointers for A, E, B, C and their corresponding Buffer objects,
 * - transpose flags for A and B,
 * - matrix dimensions M, N, K,
 * - warp allocation policy and clear_accum flag,
 * - strides and memory offsets for A and B,
 * - optional kPack (must be 1 or 2) and optional wg_wait.
 *
 * The populated GemmSPPyNode is stored into the wrapper's internal `data_`.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Eptr, Bptr, Cptr, trans_A (Bool), trans_B (Bool),
 *      M (Int), N (Int), K (Int), policy (Int), clear_accum (Bool),
 *      stride_A (Int), stride_B (Int), offset_A (Int), offset_B (Int),
 *      (optional) kPack (Int), (optional) wg_wait (Int)]
 * @param vmap Mapping from access pointer vars to Buffer objects used to
 *   resolve the Buffer corresponding to each pointer argument.
 *
 * @note If `kPack` is provided it must be 1 or 2; otherwise the constructor
 *       fails with an ICHECK (runtime assertion). No other validation is
 *       performed here.
 */
GemmSPPy::GemmSPPy(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<GemmSPPyNode> node = tvm::ffi::make_object<GemmSPPyNode>();

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

AccessRegions GemmSPPyNode::GetAccessRegions() const {
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
 * @brief Create a copy of this GemmSPPyNode as a TileOperator.
 *
 * Constructs a new GemmSPPyNode by copying the current node state and returns
 * it wrapped in a GemmSPPy TileOperator.
 *
 * @return TileOperator A GemmSPPy operator that owns a copy of this node.
 */
TileOperator GemmSPPyNode::Clone() const {
  auto op = tvm::ffi::make_object<GemmSPPyNode>(*this);
  return GemmSPPy(op);
}

Stmt GemmSPPyNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  if (const auto f = ffi::Function::GetGlobal("tl.gemm_sp_py.lower")) {
    auto prim_func =
        Downcast<PrimFunc>((*f)(tvm::ffi::GetRef<GemmSPPy>(this), T.target,
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
    // warp with block realize node
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
    LOG(FATAL) << "No lower function found for gemm_sp_py";
  }
}

LayoutMap GemmSPPyNode::InferLayout(const LayoutInferArgs &T,
                                    InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;

  if (const auto f = ffi::Function::GetGlobal("tl.gemm_sp_py.infer_layout")) {
    results = Downcast<LayoutMap>(
        (*f)(tvm::ffi::GetRef<GemmSPPy>(this), T.target, T.thread_bounds));
  } else {
    LOG(FATAL) << "No infer layout function found for gemm_sp_py";
  }

  completed_ = true;
  return results;
}

TIR_REGISTER_TL_TILE_OP(GemmSPPy, gemm_sp_py)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { GemmSPPyNode::RegisterReflection(); }
} // namespace tl
} // namespace tvm
