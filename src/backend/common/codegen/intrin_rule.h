/*!
 * \file backend/common/codegen/intrin_rule.h
 * \brief Native intrinsic rewriting shared by TileLang source backends.
 */
#ifndef TILELANG_BACKEND_COMMON_CODEGEN_INTRIN_RULE_H_
#define TILELANG_BACKEND_COMMON_CODEGEN_INTRIN_RULE_H_

#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>

#include <string>

namespace tvm::tl::intrin {

inline PrimExpr CallPureExtern(const CallObj *call, const std::string &name,
                               ffi::Array<Expr> operands) {
  ffi::Array<Expr> args{StringImm(name)};
  for (const auto &operand : operands)
    args.push_back(operand);
  return Call(call->ty, tirx::call_pure_extern_op(), std::move(args),
              call->attrs, {}, call->loc)
      .as_or_throw<PrimExpr>();
}

// Some backends pack vectors in integer carriers. The extern still has a
// vector result type; their emitter calls the scalar function once per lane.
template <typename Mapper, bool type_from_arg = false,
          bool scalarize_name = false>
PrimExpr DispatchPureExtern(const PrimExpr &expr) {
  const auto *call = expr.as<CallObj>();
  TVM_FFI_ICHECK(call != nullptr);
  const Op op = call->op.as_or_throw<Op>();
  TVM_FFI_ICHECK_EQ(op->name.substr(0, 5), "prim.");
  if constexpr (type_from_arg)
    TVM_FFI_ICHECK_EQ(call->args.size(), 1);
  PrimType type =
      type_from_arg ? call->args[0].as_or_throw<PrimExpr>().ty() : expr.ty();
  if constexpr (scalarize_name)
    type = type.WithLanes(1);
  const auto name = Mapper()(type, op->name.substr(5));
  return name.empty() ? expr : CallPureExtern(call, name, call->args);
}

} // namespace tvm::tl::intrin
#endif // TILELANG_BACKEND_COMMON_CODEGEN_INTRIN_RULE_H_
