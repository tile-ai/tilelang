#ifndef TILELANG_TRANSFORM_REDUCER_METADATA_H_
#define TILELANG_TRANSFORM_REDUCER_METADATA_H_

#include "ir/annotations.h"
#include "ir/lexical_scope.h"
#include "op/reducer.h"
#include "op/utils.h"
#include <functional>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/tirx/stmt_functor.h>

namespace tvm::tl::reducer {
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;
using tirx::TensorVar;
using Metadata = Map<String, Any>;
using InfoMap = Map<TensorVar, Metadata>;

inline Metadata StringMap(const Any &raw) {
  Metadata result;
  for (const auto &[key, value] : raw.as_or_throw<Map<Any, Any>>())
    result.Set(key.as_or_throw<String>(), value);
  return result;
}

inline TensorVar TensorKey(const Any &raw) {
  Expr expression = raw.as_or_throw<Expr>();
  if (const auto *call = expression.as<CallObj>();
      call && call->op.same_as(tirx::tensor_data_ptr_op()))
    expression = call->args[0];
  return expression.as_or_throw<TensorVar>();
}

inline InfoMap ReadInfo(const Any &raw) {
  InfoMap result;
  for (const auto &[key, value] : raw.as_or_throw<Map<Any, Any>>())
    result.Set(TensorKey(key), StringMap(value));
  return result;
}

inline Map<TensorVar, Layout> ReadLayouts(const Any &raw) {
  Map<TensorVar, Layout> result;
  for (const auto &[key, value] : raw.as_or_throw<Map<Any, Any>>())
    result.Set(TensorKey(key), value.as_or_throw<Layout>());
  return result;
}

inline bool IsMetadataCall(const CallObj *call) {
  return IsAnnotationCall(call);
}

inline bool IsReducerMetadataKey(const String &key) {
  return key == attr::kReducerInfo || key == attr::kReducerInfoV2 ||
         key == attr::kLayoutMap;
}

// Metadata references are data contracts, not executable reducer accesses.
// Collect them separately before walking epoch execution in statement order.
struct MetadataIndex {
  InfoMap legacy;
  InfoMap v2;
  Map<TensorVar, Layout> layouts;
  Map<TensorVar, Bind> allocations;

  void Record(const Metadata &metadata) {
    for (const auto &[key, raw] : metadata) {
      if (key == attr::kReducerInfo || key == attr::kReducerInfoV2) {
        auto &target = key == attr::kReducerInfo ? legacy : v2;
        for (const auto &[tensor, info] : ReadInfo(raw)) {
          auto existing = target.Get(tensor);
          TVM_FFI_CHECK(!existing || ffi::StructuralEqual()(*existing, info),
                        ValueError)
              << "Conflicting reducer metadata for " << tensor;
          target.Set(tensor, info);
        }
      } else if (key == attr::kLayoutMap) {
        for (const auto &[tensor, layout] : ReadLayouts(raw))
          layouts.Set(tensor, layout);
      }
    }
  }

  static MetadataIndex Collect(const tirx::Function &function) {
    MetadataIndex index;
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return index;
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        function->body,
        [&](const ffi::ObjectRef &node) -> ffi::Expected<ffi::WalkResult> {
          if (const auto *bind = node.as<BindObj>()) {
            if (auto tensor = bind->var.as<TensorVar>()) {
              const auto *call = bind->value.as<CallObj>();
              if (call && call->op.same_as(tirx::alloc_tensor_op()))
                index.allocations.Set(*tensor, ffi::GetRef<Bind>(bind));
            }
          } else if (const auto *call = node.as<CallObj>()) {
            if (IsMetadataCall(call))
              index.Record(ReadAnnotations(call));
          } else if (const auto *region = node.as<RegionStmtObj>();
                     region && region->op->name == "tl.attr") {
            auto fields = ReadAnnotations(region);
            auto key = fields.at("key").as_or_throw<String>();
            if (key == "tl.kernel")
              index.Record(StringMap(fields.at("value")));
            else if (IsReducerMetadataKey(key))
              index.Record({{key, fields.at("value")}});
          }
          return ffi::WalkResult::Advance();
        });
    return index;
  }
};

// Applied after executable statements have been rewritten, so all produced
// physical layouts are known. Every copy of an inferred layout map receives
// the same override; stale nested maps cannot shadow the committed plan.
class MetadataRewriter : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  using Rewrite = std::function<Metadata(Metadata)>;
  explicit MetadataRewriter(Rewrite rewrite) : rewrite_(std::move(rewrite)) {}

  ffi::UnchangedOr<Stmt> Mutate_(const EvaluateObj *op,
                                 InplaceMode mode) final {
    if (const auto *call = op->value.as<CallObj>();
        call && IsMetadataCall(call)) {
      auto metadata = rewrite_(ReadAnnotations(call));
      if (metadata.empty())
        return Evaluate(0, op->loc);
      return MakeAnnotation(metadata, op->loc);
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (op->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(op, mode);
    auto fields = ReadAnnotations(op);
    const String key = fields.at("key").as_or_throw<String>();
    SeqStmt body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    if (key == "tl.kernel") {
      return MakeAttribute(fields.at("node"), key,
                           rewrite_(StringMap(fields.at("value"))), body,
                           op->loc);
    }
    if (IsReducerMetadataKey(key)) {
      auto metadata = rewrite_({{key, fields.at("value")}});
      Array<Stmt> statements;
      if (!metadata.empty())
        statements.push_back(MakeAnnotation(metadata, op->loc));
      statements.push_back(body);
      return RegionStmt(tl::lexical_scope_op(), {}, {}, DictAttrs(),
                        SeqStmt(statements, op->loc), {}, op->loc);
    }
    return RegionStmt(op->op, op->args, op->body_params, op->attrs, body,
                      op->result_vars, op->loc);
  }

private:
  Rewrite rewrite_;
};

} // namespace tvm::tl::reducer
#endif // TILELANG_TRANSFORM_REDUCER_METADATA_H_
