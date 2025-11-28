/*!
 * \file tl/op/op.cc
 *
 * Define operators usd in tile library.
 */

#include "operator.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tir;

/**
 * @brief Construct a TileOperator from a TIR Call using a registered builder.
 *
 * Looks up a builder function in the "TLOpBuilder" Op attribute map for the
 * operator referenced by `call` and invokes it to produce a TileOperator. If no
 * builder is registered for the operator, returns a default-constructed (empty)
 * TileOperator.
 *
 * @param call The TIR Call whose operator and arguments will be used to build
 * the TileOperator.
 * @return TileOperator The constructed TileOperator, or a default (empty)
 * TileOperator if no builder exists.
 */
TileOperator ParseOperator(Call call) {
  auto op_map = Op::GetAttrMap<OpBuilderFunc>("TLOpBuilder");
  Op op = call->op.as<Op>().value();
  if (op_map.count(op)) {
    auto tile_op = op_map[op](call->args);
    ICHECK(tile_op.defined());
    return tile_op;
  }
  return TileOperator();
}

/**
 * @brief Parse a TileOperator from a TIR statement if it contains a call.
 *
 * If `stmt` is an Evaluate node whose value is a Call, delegates to
 * ParseOperator(Call, BufferMap) and returns the resulting TileOperator.
 * Otherwise returns a default-constructed (empty) TileOperator.
 *
 * @param stmt TIR statement to inspect; expected to be an Evaluate of a Call.
 * @return TileOperator Parsed operator on success, or a default (empty)
 * TileOperator if `stmt` is not an Evaluate(Call).
 */
TileOperator ParseOperator(Stmt stmt) {
  if (stmt.as<Evaluate>() && stmt.as<EvaluateNode>()->value.as<CallNode>()) {
    auto call = stmt.as<EvaluateNode>()->value.as<CallNode>();
    return ParseOperator(tvm::ffi::GetRef<Call>(call));
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
  ICHECK(call->op.same_as(builtin::tvm_access_ptr()));
  auto var = call->args[1].as<VarNode>();
  ICHECK(var);
  return tvm::ffi::GetRef<Var>(var);
}

} // namespace tl
} // namespace tvm
