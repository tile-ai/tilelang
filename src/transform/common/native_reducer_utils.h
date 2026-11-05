/*!
 * \file transform/common/native_reducer_utils.h
 * \brief Native tensor definitions and explicit reducer metadata utilities.
 */
#ifndef TILELANG_TRANSFORM_COMMON_NATIVE_REDUCER_UTILS_H_
#define TILELANG_TRANSFORM_COMMON_NATIVE_REDUCER_UTILS_H_

#include "ir/lexical_scope.h"
#include "native_layout_utils.h"
#include "op/reducer.h"
#include "op/utils.h"

#include <functional>
#include <unordered_map>

namespace tvm::tl::reducer_native {
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
  const auto *op = call->op.as<OpObj>();
  return op && op->name == "tl.annotation";
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
    VisitNativePostOrder(function->body, [&](const ffi::ObjectRef &node) {
      if (const auto *bind = node.as<BindObj>()) {
        if (auto tensor = bind->var.as<TensorVar>()) {
          const auto *call = bind->value.as<CallObj>();
          if (call && call->op.same_as(tirx::alloc_tensor_op()))
            index.allocations.Set(*tensor, ffi::GetRef<Bind>(bind));
        }
      } else if (const auto *call = node.as<CallObj>()) {
        if (IsMetadataCall(call))
          index.Record(ReadNativeAnnotations(call));
      } else if (const auto *region = node.as<RegionStmtObj>();
                 region && region->op->name == "tl.attr") {
        auto fields = ReadNativeAnnotations(region);
        auto key = fields.at("key").as_or_throw<String>();
        if (key == "tl.kernel")
          index.Record(StringMap(fields.at("value")));
        else if (IsReducerMetadataKey(key))
          index.Record({{key, fields.at("value")}});
      }
    });
    return index;
  }
};

inline TensorVar RegionTensor(const Expr &operand) {
  return tl::RegionTensor(NormalizeToTensorRegion(operand));
}

inline TensorRegion FullRegion(const TensorVar &tensor) {
  return FullTensorRegion(tensor);
}

inline TensorVar WithScope(const TensorVar &tensor, const char *scope,
                           const String &suffix = "") {
  auto type = ffi::make_object<tirx::TensorTypeObj>(*tensor.ty().get());
  type->scope = runtime::MemoryScope::Get(scope);
  return TensorVar(tensor->name + suffix, tirx::TensorType(type), tensor->loc);
}

inline Stmt Allocate(const TensorVar &tensor, Location loc = UnknownLoc(),
                     ffi::Optional<Attrs> attrs = std::nullopt) {
  return Bind(
      tensor,
      Call(tensor.ty(), tirx::alloc_tensor_op(), {}, attrs, {tensor.ty()}, loc),
      loc);
}

inline Stmt TileCall(const Op &op, const Array<Expr> &operands,
                     const Metadata &metadata = {},
                     Location loc = UnknownLoc()) {
  return Evaluate(
      MakeAnnotatedCall(PrimType::Void(), op, operands, metadata, loc), loc);
}

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
      auto metadata = rewrite_(ReadNativeAnnotations(call));
      if (metadata.empty())
        return Evaluate(0, op->loc);
      return MakeNativeAnnotation(metadata, op->loc);
    }
    return StmtExprMutator::Mutate_(op, mode);
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    if (op->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(op, mode);
    auto fields = ReadNativeAnnotations(op);
    const String key = fields.at("key").as_or_throw<String>();
    SeqStmt body = Mutate(op->body, mode).ValueOrUnchanged(op->body);
    if (key == "tl.kernel") {
      return MakeNativeAttribute(fields.at("node"), key,
                                 rewrite_(StringMap(fields.at("value"))), body,
                                 op->loc);
    }
    if (IsReducerMetadataKey(key)) {
      auto metadata = rewrite_({{key, fields.at("value")}});
      Array<Stmt> statements;
      if (!metadata.empty())
        statements.push_back(MakeNativeAnnotation(metadata, op->loc));
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

} // namespace tvm::tl::reducer_native
#endif // TILELANG_TRANSFORM_COMMON_NATIVE_REDUCER_UTILS_H_
