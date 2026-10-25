#ifndef TVM_TL_TRANSFORM_COMMON_PIPELINE_REPLAY_H_
#define TVM_TL_TRANSFORM_COMMON_PIPELINE_REPLAY_H_

#include "pipeline_utils.h"
#include "read_only_params.h"

#include <tvm/tirx/function.h>

namespace tvm::tl {

using PipelineReadOnlySet = std::unordered_set<const VarNode *>;

inline PipelineReadOnlySet
PipelineReadOnlyParameters(const tirx::Function &function) {
  PipelineReadOnlySet result;
  if (function->GetAttr<bool>(tirx::attr::kNoAlias).value_or(true))
    for (const IntImm &index : ReadOnlyParameterIndices(function))
      result.insert(function->params[index->value.as<int>().value()].get());
  return result;
}

inline bool PipelineCanReplay(const PrimExpr &value,
                              const MemoryAliases &aliases,
                              const PipelineReadOnlySet &readonly) {
  if (SideEffect(value) == CallEffectKind::kPure)
    return true;
  if (SideEffect(value) > CallEffectKind::kReadState)
    return false;
  bool safe = true;
  static auto effects = Op::GetAttrMap(tvm::op_attr::kCallEffectKind);
  PostOrderVisit(value, [&](const ffi::ObjectRef &object) {
    if (const auto *load = object.as<TensorLoadNode>()) {
      auto location = aliases.Resolve(load->source);
      safe &= location.allocation && location.scope == "global" &&
              readonly.count(location.allocation.value().get());
    }
    if (const auto *call = object.as<CallNode>())
      safe &= effects.get(call->op, CallEffectKind::kOpaque) ==
              CallEffectKind::kPure;
  });
  return safe;
}

inline std::vector<int>
PipelineReplayMask(const SeqStmt &body, const MemoryAliases &aliases,
                   const PipelineReadOnlySet &readonly,
                   const ffi::Map<ffi::String, ffi::Any> &attrs,
                   bool automatic) {
  std::vector<int> requested(body->seq.size(), 0), result(body->seq.size(), 0);
  if (auto mask = attrs.Get(kPipelineReplayableScalarBinds)) {
    requested = PipelineIntegerArray(*mask, kPipelineReplayableScalarBinds);
    TVM_FFI_CHECK(requested.size() == body->seq.size(), ValueError)
        << "Pipeline scalar replay mask must match the source statements";
    for (int flag : requested)
      TVM_FFI_CHECK(flag <= 1, ValueError)
          << "Pipeline scalar replay mask must be boolean";
  }
  ffi::Map<PrimVar, PrimExpr> definitions;
  for (size_t index = 0; index < body->seq.size(); ++index) {
    const auto *binding = body->seq[index].as<BindNode>();
    auto variable = binding ? binding->var.as<PrimVar>() : std::nullopt;
    auto expression = binding ? binding->value.as<PrimExpr>() : std::nullopt;
    bool safe = false;
    bool pure = false;
    if (variable && expression) {
      PrimExpr value = Substitute(*expression, definitions);
      pure = SideEffect(value) == CallEffectKind::kPure;
      safe = PipelineCanReplay(value, aliases, readonly);
      if (safe && (pure || automatic || requested[index]))
        definitions.Set(*variable, value);
    }
    TVM_FFI_CHECK(!requested[index] || safe, ValueError)
        << "Pipeline scalar binding cannot be replayed safely";
    result[index] = safe && (pure || automatic || requested[index]);
  }
  return result;
}

inline std::vector<int> NormalizePipelineOrder(std::vector<int> order) {
  auto sorted = order;
  std::sort(sorted.begin(), sorted.end());
  for (int &value : order)
    value =
        std::lower_bound(sorted.begin(), sorted.end(), value) - sorted.begin();
  return order;
}

inline ffi::Map<ffi::String, ffi::Any>
FilterPipelineMetadata(ffi::Map<ffi::String, ffi::Any> attrs,
                       const PipelineBody &body, size_t source_count) {
  if (source_count != body.statements.size()) {
    for (const char *key :
         {kPipelineStage, kPipelineOrder, kPipelineAsyncProducers,
          kPipelineAsyncProducerGroups, kPipelineTmaCopies}) {
      auto value = attrs.Get(key);
      if (!value)
        continue;
      auto array = value->cast<ffi::Array<ffi::Any>>();
      if (array.size() != source_count)
        continue;
      ffi::Array<ffi::Any> filtered;
      for (size_t index : body.source_indices)
        filtered.push_back(array[index]);
      attrs.Set(key, filtered);
    }
  }
  return attrs;
}

} // namespace tvm::tl

#endif
