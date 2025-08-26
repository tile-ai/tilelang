/*!
 * \file tl/op/op.cc
 *
 * Define operators usd in tile library.
 */

#include "operator.h"

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tir;


std::unique_ptr<TileOperator> ParseOperator(Call call, BufferMap vmap) {
  auto op_map = Op::GetAttrMap<OpBuilderFunc>("TLOpBuilder");
  Op op = call->op.as<Op>().value();
  if (op_map.count(op)) {
    TileOperator *ptr = static_cast<TileOperator *>(op_map[op](call->args, vmap));
    ICHECK(ptr != nullptr);
    return std::unique_ptr<TileOperator>(ptr);
  }
  return nullptr;
}

std::unique_ptr<TileOperator> ParseOperator(Stmt stmt, BufferMap vmap) {
  if (stmt.as<Evaluate>() && stmt.as<EvaluateNode>()->value.as<CallNode>()) {
    auto call = stmt.as<EvaluateNode>()->value.as<CallNode>();
    return ParseOperator(GetRef<Call>(call), vmap);
  }
  return nullptr;
}

Var GetVarFromAccessPtr(const PrimExpr &expr) {
  auto call = expr.as<CallNode>();
  ICHECK(call);
  ICHECK(call->op.same_as(builtin::tvm_access_ptr()));
  auto var = call->args[1].as<VarNode>();
  ICHECK(var);
  return GetRef<Var>(var);
}

} // namespace tl
} // namespace tvm
