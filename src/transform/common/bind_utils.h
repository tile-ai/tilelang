/*!
 * \file bind_utils.h
 * \brief Shared helpers for scalar Bind analysis.
 */
#ifndef TVM_TL_TRANSFORM_COMMON_BIND_UTILS_H_
#define TVM_TL_TRANSFORM_COMMON_BIND_UTILS_H_

#include <tvm/ir/analysis.h>
#include <tvm/ir/type.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>

#include <functional>
#include <unordered_set>

namespace tvm {
namespace tl {

using namespace tirx;

using TensorSet =
    std::unordered_set<TensorVar, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

inline bool IsReplayableScalarBind(
    const Stmt &stmt, const ffi::Array<TensorRegion> &reads,
    const TensorSet &write_tensors,
    const std::function<bool(const TensorVar &, const TensorVar &)> &may_alias =
        {}) {
  const auto *bind = stmt.as<BindObj>();
  if (bind == nullptr) {
    return false;
  }
  auto var_type = bind->var->ty.as<PrimType>();
  auto value_type = bind->value->ty.as<PrimType>();
  if (!var_type || !value_type || !var_type.value().IsScalar() ||
      !value_type.value().IsScalar()) {
    return false;
  }
  // A bind value that carries a side effect (e.g. an atomic RMW returning the
  // previous value, or any opaque call) must be materialized exactly once:
  // replaying it at every use site re-executes the side effect and silently
  // corrupts results. Pure and read-only values remain replayable; read
  // consistency is guaranteed by the read/write disjointness check below.
  if (SideEffect(bind->value) > CallEffectKind::kReadState) {
    return false;
  }
  for (const TensorRegion &read : reads) {
    auto tensor = read->source.as_or_throw<TensorVar>();
    for (const TensorVar &write : write_tensors)
      if (!may_alias || may_alias(tensor, write))
        return false;
  }
  return true;
}

// Native tensor and pointer aliases have real value operands. Unlike the old
// DeclBuffer metadata, a decl_tensor may depend on the logical iteration or on
// a load used to compute its byte offset. Replay only immutable definitions
// whose value and shape reads cannot observe writes in this pipeline.
inline bool IsReplayablePipelineBind(
    const Stmt &stmt, const ffi::Array<TensorRegion> &reads,
    const TensorSet &write_tensors,
    const std::function<bool(const TensorVar &, const TensorVar &)>
        &may_alias) {
  if (IsReplayableScalarBind(stmt, reads, write_tensors, may_alias))
    return true;
  const auto *bind = stmt.as<BindObj>();
  if (!bind ||
      (!bind->var->ty.as<PtrTypeObj>() && !bind->var->ty.as<TensorTypeObj>()) ||
      SideEffect(bind->value) > CallEffectKind::kReadState)
    return false;
  if (const auto *call = bind->value.as<CallObj>();
      call && call->op.same_as(alloc_tensor_op()))
    return false;
  for (const TensorRegion &read : reads) {
    auto tensor = read->source.as_or_throw<TensorVar>();
    for (const TensorVar &write : write_tensors)
      if (!may_alias || may_alias(tensor, write))
        return false;
  }
  return true;
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_COMMON_BIND_UTILS_H_
