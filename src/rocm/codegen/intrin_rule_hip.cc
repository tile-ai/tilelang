/*!
 * \file intrin_rule_hip.cc
 * \brief HIP intrinsic rules.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/op.h>
#include <tvm/tirx/op/attr.h>
#include <tvm/tirx/op/gpu.h>

#include "backend/common/codegen/intrin_rule.h"

namespace tvm::tl {
namespace {

struct HIPMath {
  std::string operator()(const PrimType &t, std::string name) const {
    if (t.MatchesCode(kDLFloat)) {
      // Match TIR's ties-to-even constant folding, not C99 round's ties-away.
      if (name == "round" && (t.bits() == 32 || t.bits() == 64)) {
        name = "nearbyint";
      }
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16: {
        if (name == "fabs") {
          return "__habs";
        } else if (name == "round") {
          return "hrint";
        } else {
          return "h" + name;
        }
      }
      default:
        return "";
      }
    } else if (t.MatchesElementType(kDLBfloat, 16)) {
      if (name == "fabs") {
        return "__habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (t.MatchesCode(kDLInt, kDLUInt)) {
      switch (t.bits()) {
      case 32:
        return "__" + name;
      case 64:
        return "__" + name + "ll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct HIPFastMath : public HIPMath {
  std::string operator()(const PrimType &t, std::string name) const {
    if (t.MatchesCode(kDLFloat) && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return HIPMath::operator()(t, name);
    }
    return "";
  }
};

struct HIPFastMathTan : public HIPMath {
  std::string operator()(const PrimType &t, std::string name) const {
    if (t.MatchesCode(kDLFloat)) {
      switch (t.bits()) {
      case 64:
        return name;
      case 32:
        return name + 'f';
      case 16:
        return std::string("h") + name;
      default:
        return "";
      }
    }
    return "";
  }
};

struct HIPPopcount {
  std::string operator()(const PrimType &t, std::string name) const {
    if (t.MatchesCode(kDLUInt)) {
      switch (t.bits()) {
      case 32:
        return "__popc";
      case 64:
        return "__popcll";
      default:
        return "";
      }
    }
    return "";
  }
};

Type InferShuffleType(const CallObj *call) {
  return call->args[1].as_or_throw<PrimExpr>().ty();
}

PrimExpr DispatchHIPShuffle(const PrimExpr &expr) {
  const auto *call = expr.as<CallObj>();
  TVM_FFI_ICHECK(call && call->args.size() == 5);
  const std::string original = call->op.as_or_throw<Op>()->name;
  const char *name =
      original == "tirx.gpu_warp_shuffle"        ? "tirx.hip.__shfl_sync"
      : original == "tirx.gpu_warp_shuffle_up"   ? "tirx.hip.__shfl_up_sync"
      : original == "tirx.gpu_warp_shuffle_down" ? "tirx.hip.__shfl_down_sync"
                                                 : "tirx.hip.__shfl_xor_sync";
  return Call(expr.ty(), Op::Get(name),
              {call->args[0], call->args[1], call->args[2], call->args[3]},
              call->attrs, {}, call->loc)
      .as_or_throw<PrimExpr>();
}

PrimExpr DispatchHIPActiveMask(const PrimExpr &expr) {
  const auto *call = expr.as<CallObj>();
  return Call(expr.ty(), Op::Get("tirx.hip.__activemask"), {}, call->attrs, {},
              call->loc)
      .as_or_throw<PrimExpr>();
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  constexpr const char *key = "hip.FLowerIntrinsic";
  using tirx::FLowerIntrinsic;
  OpDef("prim.clz")
      .set_attr<FLowerIntrinsic>(key,
                                 intrin::DispatchPureExtern<HIPMath, true>);
  for (const char *name :
       {"floor", "ceil", "trunc", "fabs", "round", "nearbyint", "exp2", "erf",
        "cosh", "sinh", "atan", "tanh", "sqrt", "pow", "fmod"}) {
    OpDef(ffi::String("prim.") + name)
        .set_attr<FLowerIntrinsic>(
            key, intrin::DispatchPureExtern<HIPMath, false, true>);
  }
  for (const char *name :
       {"exp", "exp10", "log", "log2", "log10", "cos", "sin"}) {
    OpDef(ffi::String("prim.") + name)
        .set_attr<FLowerIntrinsic>(
            key, intrin::DispatchPureExtern<HIPFastMath, false, true>);
  }
  OpDef("prim.tan")
      .set_attr<FLowerIntrinsic>(
          key, intrin::DispatchPureExtern<HIPFastMathTan, false, true>);
  OpDef("prim.popcount")
      .set_attr<FLowerIntrinsic>(key, intrin::DispatchPureExtern<HIPPopcount>);
  for (const char *name :
       {"tirx.gpu_warp_shuffle", "tirx.gpu_warp_shuffle_up",
        "tirx.gpu_warp_shuffle_down", "tirx.gpu_warp_shuffle_xor"}) {
    OpDef(name).set_attr<FLowerIntrinsic>(key, DispatchHIPShuffle);
  }
  OpDef("tirx.gpu_warp_activemask")
      .set_attr<FLowerIntrinsic>(key, DispatchHIPActiveMask);

  // Four explicit runtime operands remain after the generic five-slot warp op.
  for (const char *name : {"__shfl_sync", "__shfl_up_sync", "__shfl_down_sync",
                           "__shfl_xor_sync"}) {
    OpDef(ffi::String("tirx.hip.") + name)
        .signature(sig::arg<IntExpr>("mask"), sig::arg<PrimExpr>("value"),
                   sig::arg<IntExpr>("lane"), sig::arg<IntExpr>("width"))
        .set_native_fn_attr<&InferShuffleType>(op_attr::kInferType)
        .set_attr(tirx::op_attr::kGlobalSymbol, ffi::String(name))
        .set_attr(op_attr::kCallEffectKind, CallEffectKind::kOpaque)
        .set_attr<bool>("hip.need_warp_shuffle", true);
  }
  OpDef("tirx.hip.__activemask")
      .set_attr(op_attr::kFixedReturnType, PrimType::UInt(32))
      .set_attr(tirx::op_attr::kGlobalSymbol, ffi::String("__activemask"))
      .set_attr(op_attr::kCallEffectKind, CallEffectKind::kOpaque)
      .set_attr<bool>("hip.need_warp_shuffle", true);
}

} // namespace tvm::tl
