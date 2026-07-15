/*!
 * \file tl/op/op.cc
 *
 * Define operators usd in tile library.
 */

#include "operator.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include "builtin.h"
#include "copy.h"

#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

void ResolveCopySafeValue(
    TileOperator tile_op,
    const ffi::Map<ffi::String, ffi::Any> &block_annotations) {
  if (!tile_op.defined() || !tile_op->IsInstance<CopyNode>()) {
    return;
  }
  auto safe_value_map = block_annotations.Get(attr::kSafeValueMap);
  if (!safe_value_map) {
    return;
  }
  auto *copy_node = const_cast<CopyNode *>(tile_op.as<CopyNode>());
  auto map = Downcast<ffi::Map<Var, PrimExpr>>(safe_value_map.value());
  if (map.count(copy_node->src->data)) {
    copy_node->src_oob_safe_value = map[copy_node->src->data];
  }
}

} // namespace

/**
 * @brief Construct a TileOperator from a TIR Call using a registered builder.
 *
 * Looks up a builder function in the "TLOpBuilder" Op attribute map for the
 * operator referenced by `call` and invokes it to produce a TileOperator. If no
 * builder is registered for the operator, returns a default-constructed (empty)
 * TileOperator.
 *
 * @param call The TIR Call whose operator and arguments will be used to build
 * the TileOperator. Its `call->annotations` are passed through as TileOp-local
 * annotations.
 * @param block_annotations The complete enclosing SBlock annotations visible
 * while parsing this TileOp.
 * @return TileOperator The constructed TileOperator, or a default (empty)
 * TileOperator if no builder exists.
 */
TileOperator ParseOperator(const Call &call) {
  return ParseOperator(call, ffi::Map<ffi::String, ffi::Any>());
}

TileOperator
ParseOperator(const Call &call,
              const ffi::Map<ffi::String, ffi::Any> &block_annotations) {
  auto op_map = Op::GetAttrMap<OpBuilderFunc>("TLOpBuilder");
  Op op = call->op.as<Op>().value();
  if (op_map.count(op)) {
    auto tile_op = op_map[op](call->args, call->annotations);
    ICHECK(tile_op.defined());
    ResolveCopySafeValue(tile_op, block_annotations);
    return tile_op;
  }
  return TileOperator();
}

/**
 * @brief Parse a TileOperator from a TIR statement if it contains a call.
 *
 * If `stmt` is an Evaluate node whose value is a Call, delegates to
 * ParseOperator(Call, block_annotations) and returns the resulting
 * TileOperator.
 * Otherwise returns a default-constructed (empty) TileOperator.
 *
 * @param stmt TIR statement to inspect; expected to be an Evaluate of a Call.
 * @return TileOperator Parsed operator on success, or a default (empty)
 * TileOperator if `stmt` is not an Evaluate(Call).
 */
TileOperator ParseOperator(const Stmt &stmt) {
  return ParseOperator(stmt, ffi::Map<ffi::String, ffi::Any>());
}

TileOperator
ParseOperator(const Stmt &stmt,
              const ffi::Map<ffi::String, ffi::Any> &block_annotations) {
  if (stmt.as<Evaluate>() && stmt.as<EvaluateNode>()->value.as<CallNode>()) {
    auto call = stmt.as<EvaluateNode>()->value.as<CallNode>();
    return ParseOperator(tvm::ffi::GetRef<Call>(call), block_annotations);
  }
  return TileOperator();
}

/**
 * @brief Extracts the Var referenced by a `tvm_access_ptr` call expression.
 *
 * The function expects `expr` to be a `Call` to the builtin `tvm_access_ptr`
 * and returns the `Var` found in the call's second argument (`args[1]`). The
 * function performs runtime checks and will abort if `expr` is not a call, the
 * call is not `tvm_access_ptr`, or the second argument is not a `Var`.
 *
 * @param expr A `PrimExpr` representing a `tvm_access_ptr(...)` call.
 * @return tvm::Var The `Var` referenced by the `tvm_access_ptr` call.
 */
Var GetVarFromAccessPtr(const PrimExpr &expr) {
  auto call = expr.as<CallNode>();
  ICHECK(call);
  if (call->op.same_as(builtin::tvm_access_ptr())) {
    auto var = call->args[1].as<VarNode>();
    ICHECK(var);
    return tvm::ffi::GetRef<Var>(var);
  }
  if (call->op.same_as(tl::access_ptr())) {
    ICHECK_EQ(call->args.size(), 3U);
    auto load = call->args[0].as<BufferLoadNode>();
    ICHECK(load);
    auto var = load->buffer->data.as<VarNode>();
    ICHECK(var);
    return tvm::ffi::GetRef<Var>(var);
  }
  LOG(FATAL) << "GetVarFromAccessPtr expects a tvm_access_ptr or tl.access_ptr "
                "call, but got: "
             << tvm::ffi::GetRef<Call>(call);
}

} // namespace tl
} // namespace tvm
