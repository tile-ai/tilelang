#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>

namespace tvm::tl::cuda {
namespace {

PrimExpr LowerPopcount(PrimExpr expression) {
  const auto *call = expression.as<CallNode>();
  TVM_FFI_ICHECK(call && call->args.size() == 1);
  PrimExpr argument = call->args[0].as_or_throw<PrimExpr>();
  PrimType type = argument.ty();
  int bits = type.bits();
  if (!type.MatchesCode(kDLInt, kDLUInt) ||
      (bits != 8 && bits != 16 && bits != 32 && bits != 64))
    return expression;
  argument = prim::cast(argument, PrimType::UInt(bits, type.lanes()));
  if (bits < 32)
    argument = prim::cast(argument, PrimType::UInt(32, type.lanes()));
  PrimExpr count =
      Call(PrimType::Int(32, type.lanes()), tirx::call_pure_extern_op(),
           {StringImm(bits == 64 ? "__popcll" : "__popc"), argument},
           call->attrs, {}, call->loc)
          .as_or_throw<PrimExpr>();
  return prim::cast(count, expression.ty());
}

PrimExpr LowerIsFinite(PrimExpr expression) {
  const auto *call = expression.as<CallNode>();
  TVM_FFI_ICHECK(call && call->args.size() == 1);
  PrimType type = call->args[0].as_or_throw<PrimExpr>().ty();
  if (!type.MatchesCode(kDLFloat) || (type.bits() != 32 && type.bits() != 64))
    return expression;
  return Call(expression.ty(), tirx::call_pure_extern_op(),
              {StringImm("isfinite"), call->args[0]}, call->attrs, {},
              call->loc)
      .as_or_throw<PrimExpr>();
}

PrimExpr LowerMinMaxNan(PrimExpr expression) {
  const auto *call = expression.as<CallNode>();
  TVM_FFI_ICHECK(call && call->args.size() == 2);
  PrimType type = expression.ty();
  TVM_FFI_CHECK(type.MatchesElementType(kDLFloat, 16) ||
                    type.MatchesElementType(kDLBfloat, 16),
                TypeError)
      << "NaN-propagating min/max requires float16 or bfloat16";
  bool maximum = call->op.as_or_throw<Op>()->name == "tl.max_nan";
  return Call(type, tirx::call_pure_extern_op(),
              {StringImm(maximum ? "__hmax_nan" : "__hmin_nan"), call->args[0],
               call->args[1]},
              call->attrs, {}, call->loc)
      .as_or_throw<PrimExpr>();
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("prim.popcount")
      .set_attr<tirx::FLowerIntrinsic>("cuda.FLowerIntrinsic", LowerPopcount,
                                       true);
  OpDef("prim.isfinite")
      .set_attr<tirx::FLowerIntrinsic>("cuda.FLowerIntrinsic", LowerIsFinite,
                                       true);
  OpDef("tl.max_nan")
      .set_attr<tirx::FLowerIntrinsic>("cuda.FLowerIntrinsic", LowerMinMaxNan);
  OpDef("tl.min_nan")
      .set_attr<tirx::FLowerIntrinsic>("cuda.FLowerIntrinsic", LowerMinMaxNan);
}

} // namespace tvm::tl::cuda
