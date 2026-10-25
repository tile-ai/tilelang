/*!
 * \file tl/op/builtin_registry.h
 * \brief Internal registration helper for TileLang intrinsic Ops.
 */

#ifndef TVM_TL_OP_BUILTIN_REGISTRY_H_
#define TVM_TL_OP_BUILTIN_REGISTRY_H_

#include <tvm/ir/op.h>
#include <tvm/tirx/op/attr.h>

#include <string>
#include <utility>

namespace tvm::tl {

template <size_t... Indices>
void SetBuiltinSignature(OpDef &definition, std::index_sequence<Indices...>) {
  definition.signature(sig::arg("arg" + std::to_string(Indices), "")...);
}

template <int NumInputs>
void RegisterBuiltin(const ffi::String &name, CallEffectKind effect,
                     bool vectorizable = false) {
  OpDef definition(name);
  if constexpr (NumInputs < 0) {
    definition.signature(sig::var_args("args", ""));
  } else {
    SetBuiltinSignature(definition, std::make_index_sequence<NumInputs>{});
  }
  definition.set_attr(tvm::op_attr::kCallEffectKind, effect);
  definition.set_attr(tvm::op_attr::kAttrsMayContainExpr, true);
  if (vectorizable) {
    definition.set_attr(tirx::op_attr::kVectorizable, true);
  }
}

} // namespace tvm::tl

#define TL_DEFINE_NAMED_BUILTIN(FunctionName, OpName, NumInputs, Effect,       \
                                Vectorizable)                                  \
  const Op &FunctionName() {                                                   \
    static const Op op = Op::Get(OpName);                                      \
    return op;                                                                 \
  }                                                                            \
  TVM_FFI_STATIC_INIT_BLOCK() {                                                \
    RegisterBuiltin<NumInputs>(OpName, CallEffectKind::Effect, Vectorizable);  \
  }

#define TIR_DEFINE_TL_BUILTIN(OpName, NumInputs, Effect, Vectorizable)         \
  TL_DEFINE_NAMED_BUILTIN(OpName, "tl." #OpName, NumInputs, Effect,            \
                          Vectorizable)

#endif // TVM_TL_OP_BUILTIN_REGISTRY_H_
