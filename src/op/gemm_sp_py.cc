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

#include "../target/utils.h"
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
GemmSPPy::GemmSPPy(Array<PrimExpr> args) {
  ObjectPtr<GemmSPPyNode> node = tvm::ffi::make_object<GemmSPPyNode>();

  node->aRegion_ = NormalizeToBufferRegion(args[0]);
  node->eRegion_ = NormalizeToBufferRegion(args[1]);
  node->bRegion_ = NormalizeToBufferRegion(args[2]);
  node->cRegion_ = NormalizeToBufferRegion(args[3]);

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

GemmInst GemmSPPyNode::GetGemmInst(int block_size, Target target) const {
  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  bool allow_wgmma = TargetIsHopper(target) && (this->M >= 64) &&
                     (num_warps % 4 == 0) && CheckWGMMA();
  if (allow_wgmma) {
    return GemmInst::kWGMMA;
  } else if (TargetIsCDNA(target)) {
    return GemmInst::kMFMA;
  } else if (TargetIsCuda(target)) {
    return GemmInst::kMMA;
  } else {
    ICHECK(0) << "Unsupported target for gemm: " << target->str();
  }
}

/**
 * @brief Checks whether WGMMA (warp-group MMA) can be used for this GEMM.
 *
 * Evaluates device-memory placement, data-type combinations, transpose flags,
 * and K divisibility constraints required for the Hopper WGMMA code path.
 *
 * The check returns true only when:
 * - B resides in shared memory ("shared" or "shared.dyn"); and
 * - (C, A, B) dtypes match one of the supported combinations below and K
 *   satisfies the required alignment; and
 * - for combinations that require specific orientations, A is not transposed
 *   and B is transposed.
 *
 * Supported combinations and constraints:
 * - C=float16:
 *   - A=float16, B=float16: K % 16 == 0
 *   - Various float8 mixes (e4m3/e5m2): require (!trans_A && trans_B) and K %
 * 32 == 0
 * - C=float32:
 *   - A=float16, B=float16: K % 16 == 0
 *   - A=bfloat16, B=bfloat16: K % 16 == 0
 *   - A=float32, B=float32: require (!trans_A && trans_B) and K % 8 == 0
 *   - Various float8 mixes: require (!trans_A && trans_B) and K % 32 == 0
 * - C=int32:
 *   - 8-bit integer combinations (Int8/UInt8): require (!trans_A && trans_B)
 * and K % 32 == 0
 *
 * @return true if WGMMA is supported for the current buffers, dtypes, and
 *         transpose/shape constraints; false otherwise.
 */
bool GemmSPPyNode::CheckWGMMA() const {
  return false; // not supported yet
  // if (B.scope() != "shared.dyn" && B.scope() != "shared") {
  //   return false;
  // }

  // if (C->dtype == DataType::Float(16)) {
  //   if (A->dtype == DataType::Float(16) && B->dtype == DataType::Float(16))
  //     return K % 16 == 0;
  //   else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e4m3())
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e5m2())
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e4m3())
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e5m2())
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else
  //     return false;
  // } else if (C->dtype == DataType::Float(32)) {
  //   if (A->dtype == DataType::Float(16) && B->dtype == DataType::Float(16))
  //     return K % 16 == 0;
  //   else if (A->dtype == DataType::BFloat(16) &&
  //            B->dtype == DataType::BFloat(16))
  //     return K % 16 == 0;
  //   else if (A->dtype == DataType::Float(32) && B->dtype ==
  //   DataType::Float(32))
  //     return (!trans_A) && trans_B && K % 8 == 0;
  //   else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e4m3())
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else if (A->dtype.is_float8_e4m3() && B->dtype.is_float8_e5m2())
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e4m3())
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else if (A->dtype.is_float8_e5m2() && B->dtype.is_float8_e5m2())
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else
  //     return false;
  // } else if (C->dtype == DataType::Int(32)) {
  //   if (A->dtype == DataType::Int(8) && B->dtype == DataType::Int(8))
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else if (A->dtype == DataType::Int(8) && B->dtype == DataType::UInt(8))
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else if (A->dtype == DataType::UInt(8) && B->dtype == DataType::Int(8))
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else if (A->dtype == DataType::UInt(8) && B->dtype == DataType::UInt(8))
  //     return (!trans_A) && trans_B && K % 32 == 0;
  //   else
  //     return false;
  // } else {
  //   return false;
  // }
}

/**
 * @brief Parse and return the numeric GPU architecture from a Target's "arch"
 * attribute.
 *
 * Examines the target's "arch" string and, if it matches the pattern
 * "sm_<num>", returns <num> as an int. If the attribute is present but does not
 * match that pattern, returns 0.
 *
 * Preconditions: the target must have an "arch" attribute (this is checked via
 * ICHECK).
 *
 * @return int The parsed architecture number (e.g., 80 for "sm_80"), or 0 if
 * the arch string does not match "sm_<num>".
 */
static int GetArchInt(Target target) {
  int arch_int = 0;
  auto s = target->GetAttr<String>("arch");
  ICHECK(s.has_value());
  std::string arch = s.value();
  if (arch.rfind("sm_", 0) == 0) {
    arch_int = std::stoi(arch.substr(3));
  } else {
    arch_int = 0;
  }
  return arch_int;
}

Stmt GemmSPPyNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto block_size = *as_const_int(T.thread_bounds->extent);
  GemmInst gemm_inst = GetGemmInst(block_size, T.target);

  auto [warp_m, warp_n] =
      policy->computeWarpPartition(M, N, block_size, T.target, gemm_inst);

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
      }
      return BlockRealize(block_realize->iter_values, block_realize->predicate,
                          block);
    }
    // warp with block realize node
    return BlockRealize(
        /*iter_values=*/Array<PrimExpr>(),
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
              /*name_hint=*/global_symbol.value(), prim_func->body));
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
