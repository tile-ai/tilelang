#include "loop_annotations.h"
#include "annotations.h"
#include <optional>
#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ir/analysis.h>
#include <unordered_set>

namespace tvm::tl {
namespace {

using Fields = ffi::Map<ffi::String, ffi::Any>;
using Variables = std::unordered_set<const VarObj *>;

DictAttrs ReencodedAttrs(const DictAttrs &attrs,
                         const EncodedAnnotations &encoded) {
  auto fields = attrs->dict;
  fields.Set("schema_version", 1);
  fields.Set("annotation_schema", encoded.schema);
  return DictAttrs(fields);
}

std::optional<Fields> MetadataRegion(const RegionStmtObj *region) {
  if (!region || region->op->name != "tl.attr")
    return std::nullopt;
  auto data = ReadAnnotations(region);
  if (data.at("key").cast<ffi::String>() != "tl.loop.annotations")
    return std::nullopt;
  TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                ValueError)
      << "Loop annotation regions cannot define parameters or results";
  return data;
}

bool IsAnnotation(const CallObj *call) {
  return call && call->op.as<OpObj>() && IsAnnotationCall(call);
}

void Merge(Fields *result, const Fields &fields) {
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

void Read(const SeqStmt &body, Fields *result) {
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
        Merge(result, ReadAnnotations(call));
    }
    // A nested For, If, assumption, or atomic task has its own execution
    // domain. Its directives never become unconditional facts of this loop.
  }
}

// The frontend encodes a multidimensional Parallel as a sole-child nest.
// Preserve that established logical-loop convention without collecting
// arbitrary child loops or sibling execution domains.
const ForObj *ParallelChild(const ForObj *loop) {
  const auto *child =
      loop->body->seq.size() == 1 ? loop->body->seq[0].as<ForObj>() : nullptr;
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
          auto before = ReadAnnotations(call);
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
      statements.push_back(MakeAnnotation(*append_, body->loc));
      changed = true;
    }
    return changed ? SeqStmt(statements, body->loc) : body;
  }

  Rewrite rewrite_;
  std::optional<Fields> append_;
  Variables local_definitions_;
};

} // namespace

// Return an analysis view of same-loop directives in lexical order. Values
// may refer to earlier body-local definitions: this map is not authorization
// to move their payloads to the For or to an outer annotation region.
ffi::Map<ffi::String, ffi::Any> ReadLoopAnnotations(const ForObj *loop) {
  ffi::Map<ffi::String, ffi::Any> result = loop->annotations;
  const ForObj *inner = loop;
  while (const auto *next = ParallelChild(inner))
    inner = next;
  Read(inner->body, &result);
  return result;
}

// Rewrite each explicit carrier at its current definition site. Static For
// attributes remain untouched; callbacks cannot introduce a body-local value
// before its definition or make a nested-scope value escape. Definitions
// outside the supplied For remain the caller's responsibility.
For RewriteLoopAnnotationCarriers(const For &loop,
                                  const LoopAnnotationRewrite &rewrite) {
  return CarrierWriter(loop, rewrite).Run(loop);
}

// Add new fields after the loop body's local definitions, inside an existing
// sole metadata wrapper when present. Out-of-scope local operands are errors.
For AppendLoopAnnotation(const For &loop,
                         const ffi::Map<ffi::String, ffi::Any> &fields) {
  if (fields.empty())
    return loop;
  return CarrierWriter(loop, {}, fields).Run(loop);
}

} // namespace tvm::tl
