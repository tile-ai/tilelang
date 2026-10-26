/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/function.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/doc_translator.h>
#include <tvm/script/printer/printer.h>

#include <algorithm>
#include <functional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tirx_compat/doc_printer.h"
#include <cctype>
#include <tvm/tirx/index_map.h>

namespace tvm::tl::ir_services {
using namespace tvm::script;
using namespace tvm::script::printer;
namespace {

ExprDoc DTypeEncodingDoc(DocTranslatorObj *translator, DLDataType type,
                         bool immediate) {
  translator->SetExtraState("tl.ir.used", true);
  auto helper =
      translator->GetOrCreateExtraState<ffi::Optional<IdDoc>>("tl.ir.alias")
          .value();
  return helper
      ->Attr(immediate ? "dtype_imm_from_dlpack" : "prim_type_from_dlpack")
      ->Call({TupleDoc(
          {LiteralDoc::Int(type.code, std::nullopt),
           LiteralDoc::Int(type.bits, std::nullopt),
           LiteralDoc::Int(static_cast<int16_t>(type.lanes), std::nullopt)})});
}

bool NeedsDTypeEncoding(DLDataType type) {
  if (type == DLDataType{kDLOpaqueHandle, 0, 0})
    return false;
  return ffi::StringToDLDataType(ffi::DLDataTypeToString(type)) != type;
}

ExprDoc IndexMapDocument(DocTranslatorObj *translator,
                         const tirx::IndexMapNode *map) {
  translator->SetExtraState("tl.ir.used", true);
  auto helper =
      translator->GetOrCreateExtraState<ffi::Optional<IdDoc>>("tl.ir.alias")
          .value();
  ExprDoc forward = translator->WithVarScope([&]() -> ExprDoc {
    ffi::Array<IdDoc> parameters;
    for (const PrimVar &variable : map->initial_indices)
      parameters.push_back(translator->VarGetOrAllocId(variable, true));
    return translator->WithExtraStateScope<bool>(
        "ir.explicit_value_type", true, [&]() -> ExprDoc {
          ffi::Array<ExprDoc> values;
          for (const PrimExpr &expression : map->final_indices)
            values.push_back(translator->Translate(expression).value());
          return LambdaDoc(parameters, TupleDoc(values));
        });
  });
  ffi::Array<ExprDoc> dtypes;
  for (const PrimVar &variable : map->initial_indices)
    dtypes.push_back(LiteralDoc::Str(
        ffi::DLDataTypeToString(variable.ty()->dtype), std::nullopt));
  ffi::Array<ffi::String> keys{"index_dtype"};
  ffi::Array<ExprDoc> values{ListDoc(dtypes)};
  if (map->inverse_index_map.has_value()) {
    keys.push_back("inverse_index_map");
    values.push_back(
        translator->Translate(map->inverse_index_map.value()).value());
  }
  return helper->Attr("index_map")->Call({forward}, keys, values);
}

struct VariableEnvironment {
  ffi::Dict<Var, IdDoc> mapped;
  ffi::Dict<Var, IdDoc> implicit_defs;
  std::unordered_set<std::string> allocated;
};

// Bindings and reservations stay stable within each function naming region.
class TranslationEngine : public DocTranslatorObj {
public:
  explicit TranslationEngine(ffi::Map<ffi::String, ffi::Any> config)
      : DocTranslatorObj(VTable(), std::move(config)) {
    for (const char *keyword :
         {"range",  "False",  "None",     "True",  "and",    "as",
          "assert", "async",  "await",    "break", "class",  "continue",
          "def",    "del",    "elif",     "else",  "except", "finally",
          "for",    "from",   "global",   "if",    "import", "in",
          "is",     "lambda", "nonlocal", "not",   "or",     "pass",
          "raise",  "return", "try",      "while", "with",   "yield"}) {
      AllocId(keyword);
    }
  }

  static IdDoc Allocate(DocTranslatorObj *d, ffi::AnyView value) {
    auto *self = static_cast<TranslationEngine *>(d);
    std::string base = value.cast<ffi::String>();
    if (base.empty())
      base = "v";
    for (char &c : base)
      if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
        c = '_';
    if (std::isdigit(static_cast<unsigned char>(base[0])))
      base.insert(base.begin(), '_');
    std::string name = base;
    auto reserved = [&](const std::string &candidate) {
      return std::any_of(
          self->var_scopes_.begin(), self->var_scopes_.end(),
          [&](const auto &frame) { return frame.allocated.count(candidate); });
    };
    for (int i = 1; reserved(name); ++i)
      name = base + '_' + std::to_string(i);
    self->var_scopes_.back().allocated.insert(name);
    return IdDoc(name);
  }

  static IdDoc GetOrAllocate(DocTranslatorObj *d, ffi::AnyView value,
                             bool explicit_def) {
    auto *self = static_cast<TranslationEngine *>(d);
    Var var = value.cast<Var>();
    for (size_t i = self->var_scopes_.size(); i != 0; --i) {
      auto &frame = self->var_scopes_[i - 1];
      auto found = frame.mapped.find(var);
      if (found == frame.mapped.end())
        continue;
      IdDoc id = (*found).second;
      if (explicit_def) {
        frame.implicit_defs.erase(var);
      }
      return id;
    }
    IdDoc id = Allocate(d, var->name);
    auto &frame = self->var_scopes_.back();
    self->RecordOrigin(id, var);
    frame.mapped.Set(var, id);
    if (!explicit_def)
      frame.implicit_defs.Set(var, id);
    return id;
  }
  static ffi::Dict<Var, IdDoc> ImplicitDefinitions(DocTranslatorObj *d) {
    return static_cast<TranslationEngine *>(d)
        ->var_scopes_.back()
        .implicit_defs;
  }
  static void BeginScope(DocTranslatorObj *d) {
    static_cast<TranslationEngine *>(d)->var_scopes_.emplace_back();
  }
  static void EndScope(DocTranslatorObj *d) {
    auto *self = static_cast<TranslationEngine *>(d);
    TVM_FFI_CHECK(self->var_scopes_.size() > 1, ValueError)
        << "printer cannot pop the root variable scope";
    auto &local = self->var_scopes_.back();
    auto &parent = self->var_scopes_[self->var_scopes_.size() - 2];
    // Unbound references remain captures of the enclosing naming region.
    for (const auto &[var, id] : local.implicit_defs) {
      parent.mapped.Set(var, id);
      parent.implicit_defs.Set(var, id);
      parent.allocated.insert(id->name);
    }
    self->var_scopes_.pop_back();
  }

private:
  template <typename Signature, auto Function> static auto Adapter() {
    using View = ffi::reflection::NativeFunctionView<Signature>;
    return reinterpret_cast<typename View::ABIType>(
        ffi::Any(View::template FromNative<Function>())
            .template cast<void *>());
  }
  static ffi::Optional<ExprDoc> Dispatch(DocTranslatorObj *d,
                                         ffi::AnyView value,
                                         const ffi::Object *destination) {
    if (auto type = value.as<PrimType>();
        type && NeedsDTypeEncoding(type.value()->dtype))
      return DTypeEncodingDoc(d, type.value()->dtype, false);
    if (auto immediate = value.as<DataTypeImm>();
        immediate && NeedsDTypeEncoding(immediate.value()->value))
      return DTypeEncodingDoc(d, immediate.value()->value, true);
    if (auto map = value.as<tirx::IndexMap>())
      return IndexMapDocument(d, map->get());
    return d->DefaultTranslate(value, destination);
  }
  static const DocTranslatorVTable *VTable() {
    static const DocTranslatorVTable table{
        Adapter<ffi::Optional<ExprDoc>(DocTranslatorObj *, ffi::AnyView,
                                       const ffi::Object *),
                &Dispatch>(),
        Adapter<IdDoc(DocTranslatorObj *, ffi::AnyView), &Allocate>(),
        Adapter<IdDoc(DocTranslatorObj *, ffi::AnyView, bool),
                &GetOrAllocate>(),
        Adapter<ffi::Dict<Var, IdDoc>(DocTranslatorObj *),
                &ImplicitDefinitions>(),
        Adapter<void(DocTranslatorObj *), &BeginScope>(),
        Adapter<void(DocTranslatorObj *), &EndScope>()};
    return &table;
  }
  /*! \brief Owned variable environments, with a persistent root scope. */
  std::vector<VariableEnvironment> var_scopes_{1};
};

Doc TranslateComplete(ffi::AnyView obj, TranslationEngine *engine,
                      const ffi::Map<ffi::String, ffi::Any> &extra_config) {
  // Reserve the same registered or explicitly configured aliases used by
  // rendering. Keep key order so overlapping aliases reserve the same generated
  // names.
  const auto &aliases = GetNamespaceAliases();
  std::vector<std::pair<ffi::String, ffi::String>> ordered(aliases.begin(),
                                                           aliases.end());
  std::sort(ordered.begin(), ordered.end());
  for (const auto &[key, fallback] : ordered) {
    if (!extra_config.count(key))
      engine->AllocId(fallback);
  }
  for (const auto &[key, value] : extra_config) {
    std::string option = key;
    if (option.size() >= 7 &&
        option.compare(option.size() - 7, 7, ".prefix") == 0)
      engine->AllocId(value.cast<ffi::String>());
  }
  // Metadata references share one allocated identifier across the completed
  // Doc.
  engine->SetExtraState("tl.ir.alias", engine->AllocId("tlir"));
  IdDoc metadata_id = engine->AllocId("metadata");
  engine->SetExtraState("ir.metadata_id", metadata_id);
  auto docs = engine->WithDocScope([&]() {
    ffi::Optional<ExprDoc> expression = engine->Translate(obj);
    if (expression.has_value()) {
      engine->Emit(ExprStmtDoc(expression.value()), obj.as<ffi::ObjectRef>());
    }
  });
  auto root_declarations = engine->WithDocScope([&]() {
    auto pending = engine->GetImplicitDefs();
    std::vector<std::pair<Var, IdDoc>> remaining(pending.begin(),
                                                 pending.end());
    for (const auto &[var, _] : remaining) {
      if (!pending.count(var))
        continue;
      if (auto rhs = engine->Translate(var, var)) {
        engine->Emit(AssignDoc(engine->VarGetOrAllocId(var, true), rhs.value(),
                               std::nullopt),
                     var);
      }
    }
  });
  if (!root_declarations.empty()) {
    for (const Doc &doc : docs)
      root_declarations.push_back(doc);
    docs = root_declarations;
  }
  ffi::Array<StmtDoc> body;
  for (const auto &stmt : ToStmtDocArray(docs))
    body.push_back(stmt);
  return StmtBlockDoc(body);
}

Doc TranslateTileLang(ffi::AnyView value,
                      ffi::Dict<Doc, ffi::ObjectRef> *origins,
                      ffi::Optional<IdDoc> *helper,
                      const ffi::Map<ffi::String, ffi::Any> &config) {
  auto engine = ffi::make_object<TranslationEngine>(config);
  Doc document = TranslateComplete(value, engine.get(), config);
  *origins = engine->GetDocOrigins();
  if (engine->GetOrCreateExtraState<bool>("tl.ir.used"))
    *helper =
        engine->GetOrCreateExtraState<ffi::Optional<IdDoc>>("tl.ir.alias");
  return document;
}

// Visit only the completed Doc graph, never the IR carried by literal values.
void VisitDocs(ffi::AnyView value,
               const std::function<void(const Doc &)> &visit,
               std::unordered_set<const ffi::Object *> *visited) {
  const ffi::Object *object = value.as<ffi::Object>();
  if (!object || !visited->insert(object).second)
    return;
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    for (const auto &item : array.value())
      VisitDocs(item, visit, visited);
  } else if (auto doc = value.as<Doc>()) {
    visit(doc.value());
    ffi::reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(object->type_index()),
        [&](const TVMFFIFieldInfo *field) {
          if (ffi::String(field->name) != "source_paths") {
            VisitDocs(ffi::reflection::FieldGetter(field)(object), visit,
                      visited);
          }
        });
  }
}

ffi::Any ReflectedField(ffi::AnyView value, const char *name) {
  ffi::Any result;
  if (const auto *object = value.as<ffi::Object>()) {
    ffi::reflection::ForEachFieldInfo(TVMFFIGetTypeInfo(object->type_index()),
                                      [&](const TVMFFIFieldInfo *field) {
                                        if (ffi::String(field->name) == name)
                                          result = ffi::reflection::FieldGetter(
                                              field)(object);
                                      });
  }
  return result;
}

// Metadata references retain their original values in the ordinary origin map.
// Recognize the existing registry[type_key][index] expression rather than a
// reserved spelling, since the generated identifier may have been renamed.
ffi::Map<ffi::String, ffi::Array<ffi::Any>>
CollectMetadata(const Doc &root, const ffi::Dict<Doc, ffi::ObjectRef> &origins,
                ffi::Optional<IdDoc> *metadata_id) {
  ffi::Map<ffi::String, ffi::Array<ffi::Any>> metadata;
  std::unordered_set<const ffi::Object *> visited;
  VisitDocs(
      root,
      [&](const Doc &doc) {
        auto reference = doc.as<IndexDoc>();
        auto original = origins.Get(doc);
        if (!reference || !original || reference.value()->indices.size() != 1)
          return;
        auto entries = reference.value()->value.as<IndexDoc>();
        if (!entries || entries.value()->indices.size() != 1)
          return;
        auto id = entries.value()->value.as<IdDoc>();
        auto key_doc = entries.value()->indices[0].as<LiteralDoc>();
        auto index_doc = reference.value()->indices[0].as<LiteralDoc>();
        if (!id || origins.count(id.value()) || !key_doc || !index_doc)
          return;
        auto key = key_doc.value()->value.as<ffi::String>();
        const auto *index = index_doc.value()->value.as<IntImmNode>();
        if (!key || key.value() != original.value()->GetTypeKey() || !index)
          return;
        auto slot = index->value.as<int64_t>();
        if (!slot.has_value() || slot.value() < 0)
          return;
        if (metadata_id->has_value() &&
            !metadata_id->value().same_as(id.value()))
          return;
        *metadata_id = id;
        auto values =
            metadata.Get(key.value()).value_or(ffi::Array<ffi::Any>{});
        while (values.size() <= static_cast<size_t>(slot.value()))
          values.push_back(nullptr);
        values.Set(slot.value(), original.value());
        metadata.Set(key.value(), values);
      },
      &visited);
  return metadata;
}

ffi::Array<ffi::String> DisplayAliases(const PrinterConfig &config) {
  ffi::Array<ffi::String> aliases;
  for (const auto &[key, fallback] : GetNamespaceAliases()) {
    aliases.push_back(config->GetExtraConfig<ffi::String>(key, fallback));
  }
  return aliases;
}

void ApplyDisplayNames(const Doc &root, const PrinterConfig &config,
                       const ffi::Dict<Doc, ffi::ObjectRef> &origins,
                       const ffi::Optional<IdDoc> &metadata_id) {
  auto block = root.as<StmtBlockDoc>();
  if (!block || block.value()->stmts.empty())
    return;
  Doc definition = block.value()->stmts.back();
  ffi::Optional<IdDoc> root_id;
  ffi::Optional<ffi::String> name;
  if (!config->binding_names.empty())
    name = config->binding_names.back();
  if (auto function = definition.as<FunctionDoc>()) {
    root_id = function.value()->name;
    if (name) {
      // Preserve a public function's declared symbol through common reflected
      // attributes, without dispatching on the function's dialect.
      if (auto original = origins.Get(definition)) {
        auto attrs =
            ReflectedField(ReflectedField(original.value(), "attrs"), "dict");
        if (auto dict = attrs.as<ffi::Map<ffi::String, ffi::Any>>()) {
          if (auto symbol = dict->Get(tvm::attr::kGlobalSymbol.name)) {
            TVM_FFI_CHECK(symbol.value().cast<ffi::String>() == name.value(),
                          TypeError)
                << "printer function global_symbol must match its definition "
                   "name";
          }
        }
      }
    }
  } else if (auto module = definition.as<ClassDoc>()) {
    root_id = module.value()->name;
    if (!name)
      name = config->GetExtraConfig<ffi::Optional<ffi::String>>(
          "ir.module_name", std::nullopt);
  }
  auto reserved = DisplayAliases(config);
  if (auto module_name = config->GetExtraConfig<ffi::Optional<ffi::String>>(
          "ir.module_name", std::nullopt)) {
    reserved.push_back(module_name.value());
  }
  if (root_id && name)
    reserved.push_back(name.value());
  std::vector<IdDoc> identifiers;
  std::unordered_set<std::string> used(reserved.begin(), reserved.end());
  std::unordered_set<const ffi::Object *> visited;
  VisitDocs(
      root,
      [&](const Doc &doc) {
        if (auto id = doc.as<IdDoc>()) {
          identifiers.push_back(id.value());
          used.insert(id.value()->name);
        }
      },
      &visited);
  for (const ffi::String &spelling : reserved) {
    std::string replacement;
    for (size_t suffix = 1;; ++suffix) {
      replacement = std::string(spelling) + "_" + std::to_string(suffix);
      if (!used.count(replacement))
        break;
    }
    bool renamed = false;
    for (const IdDoc &id : identifiers) {
      if (id->name == spelling && !(root_id && root_id.value().same_as(id)) &&
          (origins.count(id) ||
           (metadata_id && metadata_id.value().same_as(id)))) {
        id->name = replacement;
        renamed = true;
      }
    }
    if (renamed)
      used.insert(replacement);
  }
  if (root_id && name)
    root_id.value()->name = name.value();
}

bool SamePath(const AccessPath &a, const AccessPath &b) {
  return a->depth == b->depth && a->IsPrefixOf(b);
}

// Join the completed Doc tree to IR occurrences through translator-owned
// origins. Ancestor paths disambiguate a shared object used in separate
// statements; unresolved repeated uses retain every candidate path.
void RecoverDocPaths(
    ffi::AnyView value, const std::vector<AccessPath> &enclosing_ir_paths,
    const std::unordered_map<const ffi::Object *, ffi::ObjectRef> &origins,
    const std::unordered_map<const ffi::Object *, std::vector<AccessPath>>
        &ir_paths,
    std::unordered_set<const ffi::Object *> *active,
    bool function_parameter = false) {
  const ffi::Object *object = value.as<ffi::Object>();
  if (!object || !active->insert(object).second)
    return;
  std::vector<AccessPath> context = enclosing_ir_paths;
  if (auto doc = value.as<DocNode>()) {
    auto origin = origins.find(object);
    if (origin != origins.end()) {
      auto candidates = ir_paths.find(origin->second.get());
      if (candidates != ir_paths.end()) {
        context.clear();
        for (const AccessPath &path : candidates->second) {
          if (enclosing_ir_paths.empty() ||
              std::any_of(enclosing_ir_paths.begin(), enclosing_ir_paths.end(),
                          [&](const AccessPath &parent) {
                            return parent->IsPrefixOf(path);
                          })) {
            context.push_back(path);
          }
        }
        if (context.empty())
          context = candidates->second;
        // Parameter wrappers provide occurrence context; the name and type
        // carry their own visible ranges.
        if (!(function_parameter && value.as<AssignDocNode>())) {
          for (const AccessPath &path : context) {
            if (std::none_of(doc->source_paths.begin(), doc->source_paths.end(),
                             [&](const AccessPath &existing) {
                               return SamePath(existing, path);
                             })) {
              doc->source_paths.push_back(path);
            }
          }
        }
      }
    }
  }
  if (auto array = value.as<ffi::Array<ffi::Any>>()) {
    for (const auto &child : array.value()) {
      RecoverDocPaths(child, context, origins, ir_paths, active,
                      function_parameter);
    }
  } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
    for (const auto &entry : map.value()) {
      RecoverDocPaths(entry.second, context, origins, ir_paths, active);
    }
  } else {
    ffi::reflection::ForEachFieldInfo(
        TVMFFIGetTypeInfo(object->type_index()),
        [&](const TVMFFIFieldInfo *field) {
          if (ffi::String(field->name) == "source_paths")
            return;
          ffi::Any children = ffi::reflection::FieldGetter(field)(object);
          bool parameters =
              value.as<FunctionDocNode>() && ffi::String(field->name) == "args";
          if (parameters) {
            auto function_origin = origins.find(object);
            auto args = children.as<ffi::Array<AssignDoc>>();
            auto params =
                function_origin == origins.end()
                    ? std::nullopt
                    : ReflectedField(function_origin->second, "params")
                          .as<ffi::Array<ffi::Any>>();
            bool matched = args && params && args->size() == params->size();
            for (size_t i = 0; matched && i < args->size(); ++i) {
              auto origin = origins.find((*args)[i].get());
              matched = origin != origins.end() &&
                        origin->second.get() == (*params)[i].as<ffi::Object>();
            }
            if (matched) {
              // Formal argument order identifies its occurrence even when the
              // same object is also referenced by another parameter's type.
              for (size_t i = 0; i < args->size(); ++i) {
                std::vector<AccessPath> parameter_paths;
                for (const AccessPath &path : context)
                  parameter_paths.push_back(path->Attr("params")->ArrayItem(i));
                RecoverDocPaths((*args)[i], parameter_paths, origins, ir_paths,
                                active, true);
              }
              return;
            }
          }
          RecoverDocPaths(children, context, origins, ir_paths, active,
                          parameters);
        });
  }
  active->erase(object);
}

class MapDocPaths {
public:
  MapDocPaths(ffi::AnyView root, const PrinterConfig &config)
      : config_(config) {
    std::unordered_set<const ffi::Object *> ancestors;
    Index(root, AccessPath::Root(), &ancestors);
  }

  std::pair<ffi::Array<AccessPath>, ffi::Map<AccessPath, ffi::String>>
  Map(const Doc &root, const ffi::Dict<Doc, ffi::ObjectRef> &doc_origins) {
    auto underline_paths = config_->path_to_underline;
    auto annotations = config_->path_to_annotate;
    std::unordered_map<const ffi::Object *, std::vector<AccessPath>>
        paths_by_object;
    for (const auto &occurrence : occurrences_) {
      paths_by_object[occurrence.first.get()].push_back(occurrence.second);
    }
    std::unordered_map<const ffi::Object *, ffi::ObjectRef> origins;
    for (const auto &[doc, original] : doc_origins) {
      origins.insert_or_assign(doc.get(), original);
    }
    std::unordered_set<const ffi::Object *> active;
    RecoverDocPaths(root, {}, origins, paths_by_object, &active);
    // Include every requested IR occurrence, even when its own Doc is absent.
    // The renderer selects the deepest visible prefix and reports the residual
    // path for both object-based and explicit path requests.
    for (const auto &occurrence : occurrences_) {
      for (const auto &target : config_->obj_to_underline) {
        if (target.same_as(occurrence.first))
          underline_paths.push_back(occurrence.second);
      }
      for (const auto &[target, message] : config_->obj_to_annotate) {
        if (target.same_as(occurrence.first))
          annotations.Set(occurrence.second, message);
      }
    }
    return {underline_paths, annotations};
  }

private:
  void Index(ffi::AnyView value, AccessPath path,
             std::unordered_set<const ffi::Object *> *ancestors) {
    const ffi::Object *object = value.as<ffi::Object>();
    if (!object || !ancestors->insert(object).second)
      return;
    ffi::ObjectRef ref = ffi::GetRef<ffi::ObjectRef>(object);
    occurrences_.emplace_back(ref, path);
    if (auto array = value.as<ffi::Array<ffi::Any>>()) {
      for (size_t i = 0; i < array->size(); ++i)
        Index((*array)[i], path->ArrayItem(i), ancestors);
    } else if (auto map = value.as<ffi::Map<ffi::Any, ffi::Any>>()) {
      for (const auto &[key, child] : map.value())
        Index(child, path->MapItem(key), ancestors);
    } else {
      ffi::reflection::ForEachFieldInfo(
          TVMFFIGetTypeInfo(object->type_index()),
          [&](const TVMFFIFieldInfo *field) {
            Index(ffi::reflection::FieldGetter(field)(object),
                  path->Attr(ffi::String(field->name)), ancestors);
          });
    }
    ancestors->erase(object);
  }

  /*! \brief Read-only diagnostic requests for this print. */
  PrinterConfig config_;
  /*! \brief Every original object occurrence and its IR access path. */
  std::vector<std::pair<ffi::ObjectRef, AccessPath>> occurrences_;
};

ffi::String RenderScript(const ffi::ObjectRef &obj,
                         const PrinterConfig &config) {
  ffi::Dict<Doc, ffi::ObjectRef> origins;
  ffi::Optional<IdDoc> helper;
  Doc doc = TranslateTileLang(obj, &origins, &helper, config->extra_config);
  auto block = doc.as<StmtBlockDoc>();
  if (block && !config->verbose_expr && !block.value()->stmts.empty()) {
    if (auto expression = block.value()->stmts.back().as<ExprStmtDoc>()) {
      auto original = origins.Get(expression.value()->expr);
      if (original && original.value().same_as(obj)) {
        // Statement emissions retain their own input and child-expression
        // origins.
        block.value()->stmts = {block.value()->stmts.back()};
      }
    }
  }
  ffi::Optional<IdDoc> metadata_id;
  auto metadata = CollectMetadata(doc, origins, &metadata_id);
  ApplyDisplayNames(doc, config, origins, metadata_id);
  ffi::Array<ffi::Any> header;
  if (!metadata.empty()) {
    if (config->show_meta) {
      auto aliases = DisplayAliases(config);
      std::unordered_set<std::string> used(aliases.begin(), aliases.end());
      std::unordered_set<const ffi::Object *> visited;
      VisitDocs(
          doc,
          [&](const Doc &child) {
            if (auto id = child.as<IdDoc>())
              used.insert(id.value()->name);
          },
          &visited);
      std::string loader = "tlir_metadata";
      for (size_t suffix = 1; used.count(loader); ++suffix)
        loader = "tlir_metadata_" + std::to_string(suffix);
      ffi::String json =
          ffi::Function::GetGlobalRequired("tl.ir.SaveJSON")(metadata)
              .cast<ffi::String>();
      block.value()->stmts.insert(
          block.value()->stmts.begin(),
          AssignDoc(metadata_id.value(),
                    IdDoc(loader)
                        ->Attr("load_json")
                        ->Call({LiteralDoc::Str(json, std::nullopt)}),
                    std::nullopt));
      header.push_back(
          ffi::String("from tilelang._compat import ir as " + loader + "\n"));
    } else {
      block.value()->stmts.push_back(
          CommentDoc("Metadata omitted. Use show_meta=True in script() method "
                     "to show it."));
    }
  }
  bool definition = false;
  bool future_annotations = false;
  std::unordered_set<std::string> namespaces;
  std::unordered_set<const ffi::Object *> visited;
  VisitDocs(
      doc,
      [&](const Doc &child) {
        if (auto name = child.as<NamespaceDoc>())
          namespaces.insert(name.value()->canonical_name);
        if (auto function = child.as<FunctionDoc>()) {
          definition = true;
          // Defer annotations uniformly, including forward references and type
          // parameters. Header policy depends on syntax, not dialect semantics.
          future_annotations |=
              std::any_of(function.value()->args.begin(),
                          function.value()->args.end(),
                          [](const AssignDoc &arg) {
                            return arg->annotation.has_value();
                          }) ||
              function.value()->return_type.has_value() ||
              !function.value()->type_params.empty();
        }
        definition |= child.as<ClassDoc>().has_value();
      },
      &visited);
  if (future_annotations) {
    header.insert(header.begin(),
                  ffi::String("from __future__ import annotations\n\n"));
  }
  if (definition) {
    if (helper.has_value())
      header.push_back(ffi::String("from tilelang._compat import ir as " +
                                   std::string(helper.value()->name) + "\n"));
    std::vector<std::pair<ffi::String, ffi::String>> aliases(
        GetNamespaceAliases().begin(), GetNamespaceAliases().end());
    std::sort(aliases.begin(), aliases.end());
    for (const auto &[key, fallback] : aliases) {
      std::string canonical = key;
      canonical.resize(canonical.size() - std::string(".prefix").size());
      if (!namespaces.count(canonical))
        continue;
      ffi::String alias = config->GetExtraConfig<ffi::String>(key, fallback);
      ffi::String import = canonical == "s_tir" || canonical == "relax"
                               ? "from tvm." + canonical +
                                     " import script as " + std::string(alias)
                               : "from tvm_tirx.script import " + canonical +
                                     " as " + std::string(alias);
      if (config->GetExtraConfig<bool>("ir.comment_imports", false) &&
          !config->show_meta) {
        header.push_back(CommentDoc(import));
      } else {
        header.push_back(ffi::String(std::string(import) + "\n"));
      }
    }
    if (!header.empty())
      header.push_back(ffi::String("\n"));
  }
  if (config->path_to_underline.empty() && config->obj_to_underline.empty() &&
      config->path_to_annotate.empty() && config->obj_to_annotate.empty() &&
      !config->render_invisible_path_info) {
    return details::RenderPythonScript(doc, config, header, {}, {});
  }
  MapDocPaths paths(obj, config);
  auto [underline_paths, annotations] = paths.Map(doc, origins);
  return details::RenderPythonScript(doc, config, header, underline_paths,
                                     annotations);
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "tl.ir.Script", [](ffi::ObjectRef value, PrinterConfig config) {
        return RenderScript(value, config);
      });
}

} // namespace tvm::tl::ir_services
