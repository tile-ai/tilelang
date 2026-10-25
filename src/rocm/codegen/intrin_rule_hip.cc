#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/abi.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/gpu.h>

#include <string>

namespace tvm::tl::hip {
namespace {

std::string MathName(PrimType type, std::string name, bool fast) {
  if (type.MatchesCode(kDLFloat)) {
    if (name == "round" && (type.bits() == 32 || type.bits() == 64))
      name = "nearbyint";
    if (type.bits() == 64)
      return name;
    if (type.bits() == 32)
      return (fast ? "__" : "") + name + "f";
  }
  if (type.MatchesCode(kDLFloat, kDLBfloat) && type.bits() == 16) {
    if (name == "fabs")
      return "__habs";
    if (name == "round")
      return "hrint";
    return "h" + name;
  }
  if (type.MatchesCode(kDLInt, kDLUInt)) {
    if (type.bits() == 32)
      return "__" + name;
    if (type.bits() == 64)
      return "__" + name + "ll";
  }
  return "";
}

PrimExpr ScalarizedExtern(const CallNode *call, const std::string &name) {
  PrimType result_type = call->ty.as_or_throw<PrimType>();
  int lanes = result_type.lanes();
  ffi::Array<PrimExpr> results;
  ffi::Array<PrimExpr> indices;
  for (int lane = 0; lane < lanes; ++lane) {
    ffi::Array<Expr> arguments{StringImm(name)};
    for (const Expr &argument : call->args) {
      PrimExpr value = argument.as_or_throw<PrimExpr>();
      TVM_FFI_CHECK(value.ty().IsScalar() || value.ty().lanes() == lanes,
                    ValueError)
          << "HIP math operands must have matching lane counts";
      if (!value.ty().IsScalar())
        value = prim::Shuffle({value}, {IntImm::Int32(lane)});
      arguments.push_back(value);
    }
    results.push_back(Call(result_type.WithLanes(1),
                           tirx::call_pure_extern_op(), arguments, call->attrs,
                           {}, call->loc)
                          .as_or_throw<PrimExpr>());
    indices.push_back(IntImm::Int32(lane));
  }
  return lanes == 1 ? results[0] : prim::Shuffle(results, indices);
}

tirx::FLowerIntrinsic MathRule(std::string name, bool fast) {
  return [name, fast](PrimExpr expression) {
    const auto *call = expression.as<CallNode>();
    TVM_FFI_ICHECK(call);
    PrimType type = name == "clz" ? call->args[0].as_or_throw<PrimExpr>().ty()
                                  : expression.ty();
    std::string symbol;
    if (name == "popcount") {
      if (type.MatchesCode(kDLUInt) &&
          (type.bits() == 32 || type.bits() == 64)) {
        symbol = type.bits() == 32 ? "__popc" : "__popcll";
      }
    } else if (name != "tan" || type.MatchesCode(kDLFloat)) {
      symbol = MathName(type.WithLanes(1), name, fast);
    }
    return symbol.empty() ? expression : ScalarizedExtern(call, symbol);
  };
}

Type ShuffleType(const CallNode *call) {
  TVM_FFI_CHECK_EQ(call->args.size(), 4U, ValueError);
  return call->args[1]->ty;
}

void RegisterShuffle(const std::string &operation, const std::string &symbol) {
  std::string lowered = "tl.hip." + symbol;
  OpDef(lowered)
      .signature(sig::arg<IntExpr>("mask", "Thread mask"),
                 sig::arg("value", "Shuffled value"),
                 sig::arg<IntExpr>("lane", "Source lane"),
                 sig::arg<IntExpr>("width", "Group width"))
      .set_native_fn_attr<&ShuffleType>(tvm::op_attr::kInferType)
      .set_attr(tirx::op_attr::kGlobalSymbol, ffi::String(symbol))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr<bool>("hip.need_warp_shuffle", true);
  auto rule = [lowered](PrimExpr expression) {
    const auto *call = expression.as<CallNode>();
    TVM_FFI_CHECK(call && call->args.size() == 5U, ValueError)
        << "HIP shuffle requires mask, value, lane, width, and warp size";
    return Call(expression.ty(), Op::Get(lowered),
                {call->args[0], call->args[1], call->args[2], call->args[3]},
                call->attrs, {}, call->loc)
        .as_or_throw<PrimExpr>();
  };
  for (const char *backend : {"hip", "rocm"}) {
    OpDef(operation).set_attr<tirx::FLowerIntrinsic>(
        std::string(backend) + ".FLowerIntrinsic", rule, true);
  }
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  for (const char *name : {"clz", "floor", "ceil", "trunc", "fabs", "round",
                           "nearbyint", "exp2", "erf", "tan", "cosh", "sinh",
                           "atan", "tanh", "sqrt", "pow", "popcount", "fmod"}) {
    OpDef(std::string("prim.") + name)
        .set_attr<tirx::FLowerIntrinsic>("hip.FLowerIntrinsic",
                                         MathRule(name, false), true)
        .set_attr<tirx::FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                         MathRule(name, false), true);
  }
  for (const char *name :
       {"exp", "exp10", "log", "log2", "log10", "cos", "sin"}) {
    OpDef(std::string("prim.") + name)
        .set_attr<tirx::FLowerIntrinsic>("hip.FLowerIntrinsic",
                                         MathRule(name, true), true)
        .set_attr<tirx::FLowerIntrinsic>("rocm.FLowerIntrinsic",
                                         MathRule(name, true), true);
  }
  RegisterShuffle("tirx.gpu_warp_shuffle", "__shfl_sync");
  RegisterShuffle("tirx.gpu_warp_shuffle_up", "__shfl_up_sync");
  RegisterShuffle("tirx.gpu_warp_shuffle_down", "__shfl_down_sync");
  OpDef("tl.hip.__activemask")
      .signature()
      .set_attr(tvm::op_attr::kFixedReturnType, PrimType::UInt(32))
      .set_attr(tirx::op_attr::kGlobalSymbol, ffi::String("__activemask"))
      .set_attr(tvm::op_attr::kCallEffectKind, CallEffectKind::kPure)
      .set_attr<bool>("hip.need_warp_shuffle", true);
  auto active_mask = [](PrimExpr expression) {
    const auto *call = expression.as<CallNode>();
    TVM_FFI_ICHECK(call && call->args.empty());
    return Call(expression.ty(), Op::Get("tl.hip.__activemask"), {},
                call->attrs, {}, call->loc)
        .as_or_throw<PrimExpr>();
  };
  for (const char *backend : {"hip", "rocm"}) {
    OpDef("tirx.gpu_warp_activemask")
        .set_attr<tirx::FLowerIntrinsic>(
            std::string(backend) + ".FLowerIntrinsic", active_mask, true);
  }
}

} // namespace tvm::tl::hip
