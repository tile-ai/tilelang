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

#include "size_var_defaults.h"

#include "../ir/annotations.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/stmt_functor.h>

#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace tvm::tl {
namespace {

constexpr const char *kSizeVarDefaults = "tl.size_var_defaults";

bool IsIntegerScalar(const Var &variable) {
  const auto type = variable->ty.as<PrimType>();
  return type && type->lanes() == 1 && type->MatchesCode(kDLInt, kDLUInt);
}

class SourceRegistry {
public:
  void Mark(const Var &variable) {
    std::lock_guard<std::mutex> lock(mutex_);
    // Reclaim dead control blocks. Weak references keep object identity valid
    // without retaining a Var or its type graph after source construction.
    for (auto it = entries_.begin(); it != entries_.end();) {
      if (it->second.expired())
        it = entries_.erase(it);
      else
        ++it;
    }
    entries_[variable.get()] = ffi::WeakObjectPtr<VarObj>(
        ffi::GetObjectPtr<VarObj>(const_cast<VarObj *>(variable.get())));
  }

  bool Contains(const Var &variable) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(variable.get());
    return it != entries_.end() && !it->second.expired();
  }

private:
  std::mutex mutex_;
  std::unordered_map<const VarObj *, ffi::WeakObjectPtr<VarObj>> entries_;
};

SourceRegistry &Sources() {
  static SourceRegistry registry;
  return registry;
}

ffi::Array<Var> SignatureVariables(const tirx::Function &function) {
  ffi::Array<Var> variables;
  std::unordered_set<const VarObj *> seen;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      function->params,
      [&](const VarObj *node) -> ffi::Expected<ffi::WalkResult> {
        if (seen.insert(node).second)
          variables.push_back(ffi::GetRef<Var>(node));
        // TensorType shapes/layouts can themselves contain implicit Vars.
        return ffi::WalkResult::Advance();
      });
  return variables;
}

ffi::Optional<ffi::Array<ffi::Any>>
DefaultPayload(const RegionStmtObj *region) {
  if (region->op->name != "tl.attr")
    return std::nullopt;
  const auto fields =
      DecodeAnnotations(
          region->args[0].as_or_throw<Tuple>(),
          region->attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema")
              .value())
          .cast<ffi::Map<ffi::String, ffi::Any>>();
  if (fields.at("key").cast<ffi::String>() != kSizeVarDefaults)
    return std::nullopt;
  TVM_FFI_CHECK(fields.at("node") == nullptr, ValueError)
      << "Size variable defaults have no separate node payload";
  return fields.at("value").cast<ffi::Array<ffi::Any>>();
}

class DefaultStripper : public tirx::StmtExprMutator {
public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  ffi::UnchangedOr<Stmt> Mutate_(const RegionStmtObj *node,
                                 InplaceMode mode) final {
    if (!DefaultPayload(node))
      return StmtExprMutator::Mutate_(node, mode);
    TVM_FFI_CHECK(node->body_params.empty() && node->result_vars.empty(),
                  ValueError)
        << "Size variable defaults cannot define region parameters/results";
    return Mutate(node->body, mode).ValueOrUnchanged(node->body);
  }
};

} // namespace

Var MarkSizeVarDefault(Var variable) {
  TVM_FFI_CHECK(IsIntegerScalar(variable), TypeError)
      << "Only scalar integer source Vars have a size default";
  Sources().Mark(variable);
  return variable;
}

bool HasSizeVarDefault(const Var &variable) {
  return IsIntegerScalar(variable) && Sources().Contains(variable);
}

ffi::Array<Var> CollectSizeVarDefaults(const tirx::Function &function) {
  ffi::Array<Var> defaults;
  std::unordered_set<const VarObj *> seen;
  ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(
      function->body,
      [&](const RegionStmtObj *region) -> ffi::Expected<ffi::WalkResult> {
        if (auto payload = DefaultPayload(region)) {
          for (const auto &entry : *payload) {
            if (auto variable = entry.as<Var>()) {
              TVM_FFI_CHECK(IsIntegerScalar(*variable), TypeError)
                  << "Size variable default operand must be an integer scalar";
              if (seen.insert(variable->get()).second)
                defaults.push_back(*variable);
            } else {
              // Specialization may replace a marked Var by an exact value or
              // expression. It must not mark the constituent Vars of that
              // expression as nonnegative. The original shape checks remain.
              TVM_FFI_CHECK(entry.as<PrimExpr>().has_value(), TypeError)
                  << "Malformed size variable default operand";
            }
          }
        }
        return ffi::WalkResult::Advance();
      },
      [](const LambdaExprObj *) -> ffi::Expected<ffi::WalkResult> {
        return ffi::WalkResult::Skip();
      });
  return defaults;
}

tirx::Function StripSizeVarDefaults(tirx::Function function) {
  auto stripper = ffi::make_object<DefaultStripper>();
  const auto body = function->body;
  auto rewritten = stripper->Mutate(body).ValueOrUnchanged(body);
  if (!rewritten.same_as(body))
    function.CopyOnWrite()->body = rewritten;
  return function;
}

tirx::Function WithSizeVarDefaults(tirx::Function function,
                                   const ffi::Array<Var> &defaults) {
  if (function->GetAttr(tvm::attr::kDeclaration).value_or(false))
    return function;
  function = StripSizeVarDefaults(std::move(function));
  if (defaults.empty())
    return function;
  std::unordered_set<const VarObj *> requested;
  for (const Var &variable : defaults)
    requested.insert(variable.get());
  ffi::Array<Var> retained;
  for (const Var &variable : SignatureVariables(function)) {
    if (requested.count(variable.get())) {
      TVM_FFI_CHECK(IsIntegerScalar(variable), TypeError);
      retained.push_back(variable);
    }
  }
  if (retained.empty())
    return function;
  auto encoded = EncodeAnnotations(ffi::Map<ffi::String, ffi::Any>{
      {"node", nullptr}, {"key", kSizeVarDefaults}, {"value", retained}});
  SeqStmt body(RegionStmt(
      Op::Get("tl.attr"), {encoded.payload}, {},
      DictAttrs({{"schema_version", 1}, {"annotation_schema", encoded.schema}}),
      function->body, {}, function->loc));
  function.CopyOnWrite()->body = body;
  return function;
}

tirx::Function AttachSizeVarDefaults(tirx::Function function) {
  auto defaults = CollectSizeVarDefaults(function);
  for (const Var &variable : SignatureVariables(function)) {
    if (HasSizeVarDefault(variable))
      defaults.push_back(variable);
  }
  return WithSizeVarDefaults(std::move(function), defaults);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.MarkSizeVarDefault", MarkSizeVarDefault)
      .def("tl.HasSizeVarDefault", HasSizeVarDefault)
      .def("tl.CollectSizeVarDefaults", CollectSizeVarDefaults)
      .def("tl.AttachSizeVarDefaults", AttachSizeVarDefaults)
      .def("tl.WithSizeVarDefaults", WithSizeVarDefaults)
      .def("tl.StripSizeVarDefaults", StripSizeVarDefaults);
}

} // namespace tvm::tl
