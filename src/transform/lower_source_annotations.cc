/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file lower_source_annotations.cc
 * \brief Lower source annotation syntax while preserving compiler contracts.
 */

#include "ir/annotations.h"
#include "ir/lexical_scope.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <vector>

namespace tvm::tl {
namespace {
using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

Map<String, Any> ToStringMap(const Any &input) {
  Map<String, Any> result;
  for (const auto &[key, value] : input.cast<Map<Any, Any>>())
    result.Set(key.cast<String>(), value);
  return result;
}

Map<String, Any> DecodeMetadata(const CallObj *call) {
  return ReadAnnotations(call);
}

Stmt Attribute(const Any &node, const String &key, const Any &value,
               SeqStmt body, Location loc) {
  auto encoded = EncodeAnnotations(
      Map<String, Any>{{"node", node}, {"key", key}, {"value", value}});
  return RegionStmt(
      Op::Get("tl.attr"), {encoded.payload}, {},
      DictAttrs({{"schema_version", 1}, {"annotation_schema", encoded.schema}}),
      body, {}, loc);
}

// A source shell also defines an allocation and binding scope. Preserve that
// scope even when its descriptive metadata disappears; SeqStmt would flatten
// adjacent scopes and change the lifetime of their definitions.
Stmt Lexical(SeqStmt body, Location loc, bool explicit_alloc_scope = false) {
  return MakeLexicalScope(std::move(body), std::move(loc),
                          explicit_alloc_scope);
}

class SourceAnnotationLowerer : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  ffi::UnchangedOr<Stmt> Mutate_(const SeqStmtObj *op, InplaceMode mode) final {
    // Traverse definitions and operands in native order before rewriting the
    // source declarations into regions over their lexical suffix.
    auto sequence = StmtExprMutator::Mutate_(op, mode)
                        .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                        .as_or_throw<SeqStmt>();
    SeqStmt suffix(Array<Stmt>{}, sequence->loc);
    for (size_t i = sequence->seq.size(); i-- > 0;) {
      const Stmt &child = sequence->seq[i];
      const auto *evaluate = child.as<EvaluateObj>();
      const auto *call = evaluate ? evaluate->value.as<CallObj>() : nullptr;
      if (call && IsAnnotationCall(call))
        suffix = Apply(DecodeMetadata(call), suffix, child->loc);
      else
        suffix = SeqStmt({child, suffix}, sequence->loc);
    }
    return suffix;
  }

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *op,
                                 InplaceMode mode) final {
    // Native regions keep their attrs, operands, parameters and results. This
    // includes backend execution blocks, whose validation belongs to owners.
    if (op->op->name != "tl.attr")
      return StmtExprMutator::Mutate_(op, mode);
    auto region = StmtExprMutator::Mutate_(op, mode)
                      .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                      .as_or_throw<RegionStmt>();
    const auto fields = ToStringMap(DecodeAnnotations(
        region->args[0].as_or_throw<Tuple>(),
        region->attrs.GetAttr<Array<Any>>("annotation_schema").value()));
    const String key = fields.at("key").cast<String>();
    if (key == "lexical_alloc_scope")
      return Lexical(region->body, region->loc, true);
    if (key == "tl.sblock") {
      const auto block = ToStringMap(fields.at("value"));
      for (const auto &[name, value] : block) {
        TVM_FFI_CHECK(name == "name" || name == "no_realize" ||
                          name == "annotations",
                      ValueError)
            << "Unconsumed structured block field `" << name << "`";
      }
      Map<String, Any> annotations;
      if (auto values = block.Get("annotations"))
        annotations = ToStringMap(*values);
      // Axes, initialization and predicates are already native statements.
      // A source block's name never selects a backend execution domain here.
      return SourceScope(annotations, region->body, region->loc);
    }
    if (key == "tl.kernel")
      return SourceScope(ToStringMap(fields.at("value")), region->body,
                         region->loc);
    if (IsCodegenAnnotation(key) || IsCompilerAnnotation(key))
      return region;
    return Lexical(region->body, region->loc);
  }

  ffi::UnchangedOr<Expr> Mutate_(const CallObj *op, InplaceMode mode) final {
    auto call = StmtExprMutator::Mutate_(op, mode)
                    .ValueOrUnchanged(ffi::GetRef<Expr>(op))
                    .as_or_throw<Call>();
    return NormalizeCallAnnotations(std::move(call));
  }

private:
  Stmt SourceScope(Map<String, Any> values, SeqStmt body, Location loc) {
    // The header applies to this scope itself, while a standalone declaration
    // applies to the following sequence. Presence selects an explicit pool.
    const bool explicit_alloc_scope = values.count("lexical_alloc_scope");
    values.erase("lexical_alloc_scope");
    return Lexical(Apply(values, body, loc), loc, explicit_alloc_scope);
  }

  SeqStmt Apply(const Map<String, Any> &values, SeqStmt body, Location loc) {
    Map<String, Any> retained;
    std::vector<String> codegen_keys;
    bool explicit_alloc_scope = false;
    for (const auto &[key, value] : values) {
      if (key == "tl.source_block_annotation") {
        TVM_FFI_CHECK(value.as<bool>().value_or(false), ValueError)
            << "tl.source_block_annotation must be the source annotation "
               "marker";
        continue;
      }
      // Source access descriptions and axis declarations have no remaining
      // execution effect once their native definitions and accesses exist.
      if (key == "tl.axis" || key == "tl.reads" || key == "tl.writes")
        continue;
      if (key == "lexical_alloc_scope") {
        explicit_alloc_scope = true;
        continue;
      }
      if (IsCodegenAnnotation(key))
        codegen_keys.push_back(key);
      else if (IsCompilerAnnotation(key))
        retained.Set(key, value);
      // Unknown user hints disappear without evaluating their payload.
    }
    std::sort(codegen_keys.begin(), codegen_keys.end());
    for (auto it = codegen_keys.rbegin(); it != codegen_keys.rend(); ++it)
      body = SeqStmt(Attribute(Any(), *it, values.at(*it), body, loc));
    if (!retained.empty()) {
      // Keep pending metadata at its definition-safe sequence position. In
      // particular, alias exceptions can refer to a preceding local view.
      auto encoded = EncodeAnnotations(retained);
      auto call =
          Call(PrimType::Void(), Op::Get("tl.annotation"), {encoded.payload},
               DictAttrs({{"schema_version", 1},
                          {"annotation_schema", encoded.schema}}),
               {}, loc);
      body = SeqStmt({Evaluate(call, loc), body}, loc);
    }
    if (explicit_alloc_scope)
      body = SeqStmt(Lexical(body, loc, true), loc);
    return body;
  }
};
} // namespace

namespace transform {
tvm::transform::Pass LowerSourceAnnotations() {
  auto pass = [](tirx::Function function, IRModule,
                 tvm::transform::PassContext) {
    if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
      return function;
    auto mutator = ffi::make_object<SourceAnnotationLowerer>();
    auto original = function->body;
    function.CopyOnWrite()->body =
        mutator->Mutate(original).ValueOrUnchanged(original);
    return function;
  };
  return tirx::transform::CreateFunctionPass(pass, 0,
                                             "tl.LowerSourceAnnotations");
}
} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def("tl.transform.LowerSourceAnnotations",
                                   transform::LowerSourceAnnotations);
}
} // namespace tvm::tl
