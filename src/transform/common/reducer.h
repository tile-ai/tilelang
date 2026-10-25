#ifndef TILELANG_TRANSFORM_COMMON_REDUCER_H_
#define TILELANG_TRANSFORM_COMMON_REDUCER_H_

#include "op/reducer.h"
#include "op/utils.h"

#include <tvm/tirx/op/memory.h>

namespace tvm::tl {
inline constexpr const char *kReducerOperation = "tl.reducer_op";
inline constexpr const char *kLegacyReducer = "tl.reducer_legacy";

inline ffi::Optional<ffi::String> ReducerOperation(const CallNode *allocation) {
  if (!allocation || !allocation->attrs)
    return std::nullopt;
  return allocation->attrs.value()
      .as_or_throw<DictAttrs>()
      .GetAttr<ffi::String>(kReducerOperation);
}

inline TensorVar ReducerTensor(const TensorVar &tensor, const ffi::String &name,
                               const ffi::String &scope) {
  return TensorVar(name,
                   TensorType(tensor.shape(), tensor.dtype(),
                              runtime::MemoryScope::Get(scope),
                              tensor.ty()->data_alignment),
                   tensor->loc);
}

inline Stmt AllocateReducerTensor(const TensorVar &tensor,
                                  DictAttrs attributes = DictAttrs()) {
  return Bind(tensor, Call(std::nullopt, tirx::alloc_tensor_op(), {},
                           attributes, {tensor.ty()}));
}

inline Stmt ReducerLayoutHint(const TensorVar &tensor, const Layout &layout) {
  return Evaluate(Call(PrimType::Void(), Op::Get("tl.layout_hint"), {tensor},
                       DictAttrs({{"layout", layout}})));
}

inline TensorVar ReducerRegionTensor(const Expr &region) {
  return NormalizeToBufferRegion(region)->source.as_or_throw<TensorVar>();
}
} // namespace tvm::tl

#endif
