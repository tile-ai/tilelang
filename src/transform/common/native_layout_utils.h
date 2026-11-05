/*!
 * \file transform/common/native_layout_utils.h
 * \brief Explicit metadata and tensor identities used by layout passes.
 */
#ifndef TILELANG_TRANSFORM_COMMON_NATIVE_LAYOUT_UTILS_H_
#define TILELANG_TRANSFORM_COMMON_NATIVE_LAYOUT_UTILS_H_

#include "ir/annotations.h"
#include "ir/tensor_utils.h"
#include "layout/expr_utils.h"
#include "op/native_utils.h"
#include <functional>
#include <optional>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ir/analysis.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/op/region.h>
#include <tvm/tirx/stmt_functor.h>
#include <unordered_set>

namespace tvm::tl {

inline ffi::Map<ffi::String, ffi::Any>
ReadNativeAnnotations(const ffi::Array<Expr> &args, const DictAttrs &attrs) {
  TVM_FFI_CHECK_EQ(args.size(), 1U, ValueError);
  TVM_FFI_CHECK_EQ(attrs.GetAttr<int64_t>("schema_version", 0).value(), 1,
                   ValueError);
  auto decoded = DecodeAnnotations(
      args[0].as_or_throw<tvm::Tuple>(),
      attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema").value());
  ffi::Map<ffi::String, ffi::Any> result;
  for (const auto &[key, value] : decoded.cast<ffi::Map<ffi::Any, ffi::Any>>())
    result.Set(key.cast<ffi::String>(), value);
  return result;
}

inline ffi::Map<ffi::String, ffi::Any>
ReadNativeAnnotations(const CallObj *call) {
  return ReadNativeAnnotations(call->args,
                               call->attrs.value().as_or_throw<DictAttrs>());
}

inline ffi::Map<ffi::String, ffi::Any>
ReadNativeAnnotations(const RegionStmtObj *region) {
  return ReadNativeAnnotations(region->args, region->attrs);
}

inline Stmt
MakeNativeAnnotation(const ffi::Map<ffi::String, ffi::Any> &annotations,
                     Location loc = UnknownLoc()) {
  auto encoded = EncodeAnnotations(annotations);
  return Evaluate(Call(PrimType::Void(), Op::Get("tl.annotation"),
                       {encoded.payload},
                       DictAttrs({{"schema_version", 1},
                                  {"annotation_schema", encoded.schema}}),
                       {}, loc),
                  loc);
}

inline RegionStmt MakeNativeAttribute(ffi::Any node, ffi::String key,
                                      ffi::Any value, SeqStmt body,
                                      Location loc = UnknownLoc()) {
  auto encoded = EncodeAnnotations(ffi::Map<ffi::String, ffi::Any>{
      {"node", node}, {"key", key}, {"value", value}});
  return RegionStmt(
      Op::Get("tl.attr"), {encoded.payload}, {},
      DictAttrs({{"schema_version", 1}, {"annotation_schema", encoded.schema}}),
      std::move(body), {}, loc);
}

// Native SeqStmt is always the body container, including singleton nests.
inline const ForObj *SingleNestedFor(const SeqStmt &body) {
  return body->seq.size() == 1 ? body->seq[0].as<ForObj>() : nullptr;
}

namespace loop_annotation_detail {

using Fields = ffi::Map<ffi::String, ffi::Any>;
using Variables = std::unordered_set<const VarObj *>;

inline DictAttrs ReencodedAttrs(const DictAttrs &attrs,
                                const EncodedAnnotations &encoded) {
  auto fields = attrs->dict;
  fields.Set("schema_version", 1);
  fields.Set("annotation_schema", encoded.schema);
  return DictAttrs(fields);
}

inline std::optional<Fields> MetadataRegion(const RegionStmtObj *region) {
  if (!region || region->op->name != "tl.attr")
    return std::nullopt;
  auto data = ReadNativeAnnotations(region);
  if (data.at("key").cast<ffi::String>() != "tl.loop.annotations")
    return std::nullopt;
  TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                ValueError)
      << "Loop annotation regions cannot define parameters or results";
  return data;
}

inline bool IsAnnotation(const CallObj *call) {
  return call && call->op.as<OpObj>() &&
         call->op.as<OpObj>()->name == "tl.annotation";
}

inline void Merge(Fields *result, const Fields &fields) {
  for (const auto &[key, value] : fields) {
    // Each normalized hint marker contributes its own local list. Concatenate
    // those lists once, in source order, without changing their IR positions.
    if (key == "conflict_hint" && result->count(key)) {
      auto hints = result->at(key).cast<ffi::Array<ffi::Any>>();
      for (const auto &hint : value.cast<ffi::Array<ffi::Any>>())
        hints.push_back(hint);
      result->Set(key, hints);
    } else {
      result->Set(key, value);
    }
  }
}

inline void Read(const SeqStmt &body, Fields *result) {
  for (const Stmt &stmt : body->seq) {
    if (auto sequence = stmt.as<SeqStmt>()) {
      Read(*sequence, result);
    } else if (const auto *region = stmt.as<RegionStmtObj>()) {
      if (auto data = MetadataRegion(region)) {
        Merge(result, data->at("value").cast<Fields>());
        Read(region->body, result);
      }
    } else if (const auto *evaluate = stmt.as<EvaluateObj>()) {
      if (const auto *call = evaluate->value.as<CallObj>(); IsAnnotation(call))
        Merge(result, ReadNativeAnnotations(call));
    }
    // A nested For, If, assumption, or atomic task has its own execution
    // domain. Its directives never become unconditional facts of this loop.
  }
}

// The frontend encodes a multidimensional Parallel as a sole-child nest.
// Preserve that established logical-loop convention without collecting
// arbitrary child loops or sibling execution domains.
inline const ForObj *ParallelChild(const ForObj *loop) {
  const auto *child = SingleNestedFor(loop->body);
  return child && loop->kind == ForKind::kParallel &&
                 child->kind == ForKind::kParallel
             ? child
             : nullptr;
}

class CarrierWriter {
public:
  using Rewrite = std::function<Fields(const Fields &)>;
  CarrierWriter(const For &loop, Rewrite rewrite,
                std::optional<Fields> append = std::nullopt)
      : rewrite_(std::move(rewrite)), append_(std::move(append)) {
    ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
        loop,
        [&](const VarObj *var,
            TVMFFIDefRegionKind kind) -> ffi::Expected<ffi::WalkResult> {
          if (kind != kTVMFFIDefRegionKindNone)
            local_definitions_.insert(var);
          return ffi::WalkResult::Skip();
        });
  }

  For Run(const For &loop, Variables available = {}) const {
    available.insert(loop->var.get());
    SeqStmt body = loop->body;
    if (const auto *child = ParallelChild(loop.get())) {
      For rewritten = Run(ffi::GetRef<For>(child), std::move(available));
      if (!rewritten.same_as(ffi::GetRef<For>(child)))
        body = SeqStmt({rewritten}, body->loc);
    } else {
      body = Write(body, std::move(available), append_.has_value());
    }
    if (body.same_as(loop->body))
      return loop;
    For result = loop;
    result.CopyOnWrite()->body = std::move(body);
    return result;
  }

private:
  void CheckScope(const Fields &fields, const Variables &available) const {
    auto encoded = EncodeAnnotations(fields);
    auto pending = tvm::UndefinedVars(encoded.payload);
    Variables checked;
    for (size_t i = 0; i < pending.size(); ++i) {
      Var var = pending[i];
      if (!checked.insert(var.get()).second)
        continue;
      TVM_FFI_CHECK(!local_definitions_.count(var.get()) ||
                        available.count(var.get()),
                    ValueError)
          << "Loop annotation references local variable " << var->name
          << " outside its dominating definition scope";
      // A newly supplied outer variable can itself have a dependent tensor
      // type. Its shape/layout must not capture a known local before that
      // local's definition, even though ordinary Var uses are IR leaves.
      for (const Var &dependency : tvm::UndefinedVars(var->ty))
        pending.push_back(dependency);
    }
  }

  SeqStmt Write(const SeqStmt &body, Variables available, bool append) const {
    ffi::Array<Stmt> statements;
    bool changed = false;
    bool append_in_region =
        append && body->seq.size() == 1 &&
        MetadataRegion(body->seq[0].as<RegionStmtObj>()).has_value();
    for (const Stmt &stmt : body->seq) {
      Stmt result = stmt;
      if (auto sequence = stmt.as<SeqStmt>()) {
        result = Write(*sequence, available, false);
      } else if (const auto *region = stmt.as<RegionStmtObj>()) {
        if (auto data = MetadataRegion(region)) {
          auto before = data->at("value").cast<Fields>();
          auto after = rewrite_ ? rewrite_(before) : before;
          CheckScope(after, available);
          SeqStmt rewritten = Write(region->body, available, append_in_region);
          bool fields_changed = !ffi::StructuralEqual()(after, before);
          if (fields_changed || !rewritten.same_as(region->body)) {
            RegionStmt replacement = ffi::GetRef<RegionStmt>(region);
            auto *writer = replacement.CopyOnWrite();
            writer->body = std::move(rewritten);
            if (fields_changed) {
              data->Set("value", after);
              auto encoded = EncodeAnnotations(*data);
              writer->args = {encoded.payload};
              writer->attrs = ReencodedAttrs(region->attrs, encoded);
            }
            result = replacement;
          }
        }
        // Only region results, not its body-local definitions, escape.
        for (const Var &var : region->result_vars)
          available.insert(var.get());
      } else if (const auto *evaluate = stmt.as<EvaluateObj>()) {
        if (const auto *call = evaluate->value.as<CallObj>();
            IsAnnotation(call)) {
          auto before = ReadNativeAnnotations(call);
          auto after = rewrite_ ? rewrite_(before) : before;
          CheckScope(after, available);
          if (!ffi::StructuralEqual()(after, before)) {
            auto encoded = EncodeAnnotations(after);
            Call replacement = ffi::GetRef<Call>(call);
            auto *writer = replacement.CopyOnWrite();
            writer->args = {encoded.payload};
            writer->attrs = ReencodedAttrs(
                call->attrs.value().as_or_throw<DictAttrs>(), encoded);
            Evaluate rewritten = ffi::GetRef<Evaluate>(evaluate);
            rewritten.CopyOnWrite()->value = replacement;
            result = rewritten;
          }
        }
      } else if (const auto *bind = stmt.as<BindObj>()) {
        available.insert(bind->var.get());
      }
      changed |= !result.same_as(stmt);
      statements.push_back(std::move(result));
    }
    if (append && !append_in_region) {
      CheckScope(*append_, available);
      statements.push_back(MakeNativeAnnotation(*append_, body->loc));
      changed = true;
    }
    return changed ? SeqStmt(statements, body->loc) : body;
  }

  Rewrite rewrite_;
  std::optional<Fields> append_;
  Variables local_definitions_;
};

} // namespace loop_annotation_detail

// Return an analysis view of same-loop directives in lexical order. Values
// may refer to earlier body-local definitions: this map is not authorization
// to move their payloads to the For or to an outer annotation region.
inline ffi::Map<ffi::String, ffi::Any> ReadLoopAnnotations(const ForObj *loop) {
  ffi::Map<ffi::String, ffi::Any> result = loop->annotations;
  const ForObj *inner = loop;
  while (const auto *next = loop_annotation_detail::ParallelChild(inner))
    inner = next;
  loop_annotation_detail::Read(inner->body, &result);
  return result;
}

// Rewrite each explicit carrier at its current definition site. Static For
// attributes remain untouched; callbacks cannot introduce a body-local value
// before its definition or make a nested-scope value escape. Definitions
// outside the supplied For remain the caller's responsibility.
inline For RewriteLoopAnnotationCarriers(
    const For &loop,
    const loop_annotation_detail::CarrierWriter::Rewrite &rewrite) {
  return loop_annotation_detail::CarrierWriter(loop, rewrite).Run(loop);
}

// Add new fields after the loop body's local definitions, inside an existing
// sole metadata wrapper when present. Out-of-scope local operands are errors.
inline For AppendLoopAnnotation(const For &loop,
                                const ffi::Map<ffi::String, ffi::Any> &fields) {
  if (fields.empty())
    return loop;
  return loop_annotation_detail::CarrierWriter(loop, {}, fields).Run(loop);
}

// A tensor is a Var definition; aliases arise from decl_tensor's explicit
// backing pointer. Record an exact pointer expression rather than guessing from
// names or dropping a nonzero byte offset. Different typed views of the same
// pointer retain their own TensorType and layout during propagation.
class NativeTensorAliases {
public:
  ffi::Map<Var, Expr> bindings;
  ffi::Map<Expr, ffi::Array<tirx::TensorVar>> groups;

  Expr Identity(const Expr &value) const {
    if (auto var = value.as<Var>()) {
      auto bound = bindings.Get(*var);
      if (bound.has_value()) {
        if (auto call = bound.value().as<Call>()) {
          if (call.value()->op.same_as(tirx::decl_tensor_op()))
            return Identity(call.value()->args[0]);
        }
        if (bound.value()->ty.as<PtrTypeObj>())
          return Identity(bound.value());
      }
      return value;
    }
    if (const auto *call = value.as<CallObj>()) {
      if (call->op.same_as(tirx::tensor_data_ptr_op()))
        return Identity(call->args[0]);
      if (call->op.same_as(tirx::reinterpret_op()) && call->ty.as<PtrTypeObj>())
        return Identity(call->args[0]);
      if (call->op.same_as(tirx::ptr_byte_offset_op()) &&
          prim::IsZero(call->args[1].as_or_throw<PrimExpr>()))
        return Identity(call->args[0]);
    }
    return value;
  }

  void Record(const tirx::TensorVar &tensor) {
    Expr key = Identity(tensor);
    auto values = groups.Get(key).value_or(ffi::Array<tirx::TensorVar>{});
    for (const auto &old : values)
      if (old.same_as(tensor))
        return;
    values.push_back(tensor);
    groups.Set(key, values);
  }

  void Collect(const tirx::Function &function) {
    for (const Var &param : function->params)
      if (auto tensor = param.as<tirx::TensorVar>())
        Record(*tensor);
    if (!(!function->GetAttr(tvm::attr::kDeclaration).value_or(false)))
      return;
    VisitNativePostOrder(function->body, [&](const ffi::ObjectRef &node) {
      if (const auto *binding = node.as<BindObj>())
        bindings.Set(binding->var, binding->value);
    });
    VisitNativePostOrder(function->body, [&](const ffi::ObjectRef &node) {
      if (auto tensor = node.as<tirx::TensorVar>())
        Record(*tensor);
    });
  }
};

} // namespace tvm::tl
#endif // TILELANG_TRANSFORM_COMMON_NATIVE_LAYOUT_UTILS_H_
