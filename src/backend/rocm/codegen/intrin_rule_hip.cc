/*!
 * \file intrin_rule_hip.cc
 * \brief HIP intrinsic rules.
 */
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {
// Add float suffix to the intrinsics, HIP fast math.
using tirx::FLowerIntrinsic;

// HIP has no vectorized fp32 math builtins: exp2f etc. are scalar
// (float exp2f(float)), with no exp2(float4)/float2 overload (only half2 has
// packed intrinsics like h2exp2). So HIPMath/FloatSuffix return no name for a
// vector dtype and DispatchPureExtern leaves the op unlowered, which the HIP
// source codegen then can't print. Resolve the extern name from the *element*
// type instead and emit a vector call_pure_extern; CodeGenTileLangHIP::
// PrintCallExtern lowers it to one scalar call per lane and stores via
// PrintVecElemStore, which is correct for HIP's packed carrier types
// (float32x8 -> ulonglong4, float16x4 -> uint2, etc.). Scalars are unchanged.
template <typename T, bool dtype_from_arg = false>
inline PrimExpr DispatchPureExternScalarized(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  DataType dtype = dtype_from_arg ? call->args[0].dtype() : call->dtype;
  if (!dtype.is_vector()) {
    return DispatchPureExtern<T, dtype_from_arg>(e);
  }
  const OpNode *op = call->op.as<OpNode>();
  ICHECK(op != nullptr);
  std::string name = T()(dtype.element_of(), op->name.substr(5));
  if (name.empty()) {
    return e;
  }
  ffi::Array<PrimExpr> new_args = {StringImm(name)};
  for (const auto &arg : call->args) {
    new_args.push_back(arg);
  }
  return Call(call->dtype, builtin::call_pure_extern(), new_args,
              call->annotations);
}

struct HIPMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
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
    } else if (t.is_bfloat16()) {
      if (name == "fabs") {
        return "__habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (t.is_int() || t.is_uint()) {
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
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return HIPMath::operator()(t, name);
    }
    return "";
  }
};

struct HIPFastMathTan : public HIPMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
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
  std::string operator()(DataType t, std::string name) const {
    if (t.is_uint()) {
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

struct HIPWarpIntrinsic {
  const Op operator()(DataType t, const Op &orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      return Op::Get("tirx.hip.__shfl_sync");
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      return Op::Get("tirx.hip.__shfl_up_sync");
    } else {
      ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      return Op::Get("tirx.hip.__shfl_down_sync");
    }
  }
};

static PrimExpr DispatchHIPWarpActiveMask(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  return Call(call->dtype, Op::Get("tirx.hip.__activemask"), {});
}

template <typename T> static PrimExpr DispatchHIPShuffle(const PrimExpr &e) {
  // NOLINTBEGIN(clang-analyzer-cplusplus.InnerPointer)
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 5); // mask, value, warp_id, width, warp_size
  ffi::Array<PrimExpr> hip_args{
      {call->args[0], call->args[1], call->args[2], call->args[3]}};
  return Call(call->dtype, T()(call->dtype, Downcast<Op>(call->op)), hip_args);
  // NOLINTEND(clang-analyzer-cplusplus.InnerPointer)
}

TVM_REGISTER_OP("tirx.clz")
    .set_attr<FLowerIntrinsic>(
        "hip.FLowerIntrinsic",
        DispatchPureExtern<HIPMath, /*dtype_from_arg=*/true>);

TVM_REGISTER_OP("tirx.floor")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.ceil")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.trunc")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.fabs")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.round")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.nearbyint")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.exp")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPFastMath>);

TVM_REGISTER_OP("tirx.exp2")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.exp10")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPFastMath>);

TVM_REGISTER_OP("tirx.erf")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.log")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPFastMath>);

TVM_REGISTER_OP("tirx.log2")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPFastMath>);

TVM_REGISTER_OP("tirx.log10")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPFastMath>);

TVM_REGISTER_OP("tirx.tan")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPFastMathTan>);

TVM_REGISTER_OP("tirx.cos")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPFastMath>);

TVM_REGISTER_OP("tirx.cosh")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.sin")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPFastMath>);

TVM_REGISTER_OP("tirx.sinh")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.atan")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.tanh")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.sqrt")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.pow")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

TVM_REGISTER_OP("tirx.popcount")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExtern<HIPPopcount>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchHIPShuffle<HIPWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchHIPShuffle<HIPWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchHIPShuffle<HIPWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_activemask")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchHIPWarpActiveMask);

TVM_REGISTER_OP("tirx.fmod")
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               DispatchPureExternScalarized<HIPMath>);

// Register low-level builtin ops.
TVM_REGISTER_OP("tirx.hip.__shfl_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("lane", "Expr", "The source thread id.")
    .add_argument("width", "Expr",
                  "The warp thread width, must be a power of 2.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<bool>("hip.need_warp_shuffle", true);

TVM_REGISTER_OP("tirx.hip.__shfl_up_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr", "The source lane id offset to be added.")
    .add_argument("width", "Expr",
                  "The warp thread width, must be a power of 2.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_up_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<bool>("hip.need_warp_shuffle", true);

TVM_REGISTER_OP("tirx.hip.__shfl_down_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr",
                  "The source lane id offset to be subtracted.")
    .add_argument("width", "Expr",
                  "The warp thread width, must be a power of 2.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_down_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<bool>("hip.need_warp_shuffle", true);

TVM_REGISTER_OP("tirx.hip.__activemask")
    .set_num_inputs(0)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__activemask")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<bool>("hip.need_warp_shuffle", true);

} // namespace intrin
} // namespace codegen
} // namespace tvm
