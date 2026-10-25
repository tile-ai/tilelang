/*!
 * \file intrin_rule_ascend.cc
 * \brief Ascend intrinsic rules for math functions.
 *
 * The default FloatSuffix lowering uses exact dtype matching
 * (t == DataType::Float(32)) which fails for vectorized types.
 * AscendMath uses t.is_float() like CUDAMath to support any lane count.
 *
 * Registered with "ascend.FLowerIntrinsic" for the split Ascend backend.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>

namespace tvm {
namespace codegen {
namespace intrin {

using tirx::FLowerIntrinsic;

struct AscendMath {
  std::string operator()(PrimType type, std::string name) const {
    if (type.MatchesCode(kDLFloat)) {
      switch (type.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16:
        return "h" + name;
      default:
        return "";
      }
    }
    return "";
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char *name :
       {"sqrt",  "rsqrt", "exp",   "exp2", "exp10", "log",   "log2",
        "log10", "log1p", "tanh",  "tan",  "sin",   "cos",   "atan",
        "atan2", "erf",   "floor", "ceil", "fabs",  "round", "trunc"}) {
    OpDef(std::string("prim.") + name)
        .set_attr<FLowerIntrinsic>(
            "ascend.FLowerIntrinsic",
            [name](PrimExpr expression) {
              const auto *call = expression.as<CallNode>();
              TVM_FFI_ICHECK(call);
              std::string symbol = AscendMath{}(expression.ty(), name);
              if (symbol.empty())
                return expression;
              ffi::Array<Expr> arguments{StringImm(symbol)};
              for (const auto &argument : call->args)
                arguments.push_back(argument);
              return Call(expression.ty(), tirx::call_pure_extern_op(),
                          arguments, call->attrs, {}, call->loc)
                  .as_or_throw<PrimExpr>();
            },
            true);
  }
}

} // namespace intrin
} // namespace codegen
} // namespace tvm
