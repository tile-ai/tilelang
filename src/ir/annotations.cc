/*!
 * \file tl/ir/annotations.cc
 * \brief Typed layout declarations and operand-based TileLang metadata.
 */

#include "annotations.h"
#include "ascend/op/builtin.h"
#include "layout/payload.h"
#include "lexical_scope.h"
#include "transform/common/warp_specialize.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/stmt.h>
#include <tvm/script/printer/support.h>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

namespace tvm::tl {

const Op &layout_map_op() {
  static const Op op = Op::Get("tl.layout_map");
  return op;
}

bool IsAnnotationCall(const CallObj *call) {
  static const Op annotation = Op::Get("tl.annotation");
  return call &&
         (call->op.same_as(layout_map_op()) || call->op.same_as(annotation));
}

ffi::Map<tirx::TensorVar, Layout> ReadLayoutMapAnnotation(const CallObj *call) {
  TVM_FFI_CHECK(call && call->op.same_as(layout_map_op()), ValueError)
      << "Expected a tl.layout_map declaration";
  TVM_FFI_CHECK(call->args.empty() && call->ty_args.empty(), ValueError)
      << "tl.layout_map has no value or type arguments";
  TVM_FFI_CHECK(call->ty.as<PrimType>().has_value() &&
                    call->ty.as_or_throw<PrimType>().IsVoid(),
                TypeError)
      << "tl.layout_map must return void";
  TVM_FFI_CHECK(call->attrs.has_value() &&
                    call->attrs.value().as<DictAttrs>().has_value(),
                TypeError)
      << "tl.layout_map requires DictAttrs";
  const auto attrs = call->attrs.value().as_or_throw<DictAttrs>();
  for (const auto &[key, value] : attrs->dict) {
    TVM_FFI_CHECK(key == "layout_map" || key == "tl.source_block_annotation",
                  ValueError)
        << "Unknown tl.layout_map attribute: " << key;
    if (key == "tl.source_block_annotation") {
      TVM_FFI_CHECK(value.as<bool>().value_or(false), ValueError)
          << "tl.source_block_annotation must be the source annotation marker";
    }
  }
  TVM_FFI_CHECK(attrs->dict.count("layout_map"), ValueError)
      << "tl.layout_map requires its layout_map attribute";
  return attrs->dict.at("layout_map").cast<ffi::Map<tirx::TensorVar, Layout>>();
}

Stmt MakeLayoutMapAnnotation(const ffi::Map<tirx::TensorVar, Layout> &layouts,
                             bool source_block, Location loc) {
  ffi::Map<ffi::String, ffi::Any> attrs{{"layout_map", layouts}};
  if (source_block)
    attrs.Set("tl.source_block_annotation", true);
  return Evaluate(
      Call(PrimType::Void(), layout_map_op(), {}, DictAttrs(attrs), {}, loc),
      loc);
}

ffi::Array<Expr> DecodeCallOperands(const CallObj *call) {
  if (!call->attrs)
    return call->args;
  const auto *dict = call->attrs.value().as<DictAttrsObj>();
  if (!dict)
    return call->args;
  const auto attrs = ffi::GetRef<DictAttrs>(dict);
  const auto count = attrs.GetAttr<int64_t>("operand_count");
  const auto schema = attrs.GetAttr<ffi::Array<ffi::Any>>("annotation_schema");
  if (!count && !schema)
    return call->args;
  TVM_FFI_CHECK(count && schema, ValueError)
      << "Intrinsic metadata requires operand_count and annotation_schema "
         "together";
  TVM_FFI_CHECK_EQ(attrs.GetAttr<int64_t>("schema_version").value_or(0), 1,
                   ValueError);
  TVM_FFI_CHECK_GE(count.value(), 0, ValueError);
  TVM_FFI_CHECK_EQ(static_cast<size_t>(count.value()) + 1, call->args.size(),
                   ValueError)
      << "Intrinsic metadata must occupy exactly one final Tuple operand";
  const auto payload = call->args.back().as_or_throw<tvm::Tuple>();
  // Resolve the schema as well: a valid Tuple with an out-of-range variable
  // reference must not slip through and disappear at code generation.
  DecodeAnnotations(payload, schema.value());
  ffi::Array<Expr> result;
  result.reserve(count.value());
  for (int64_t i = 0; i < count.value(); ++i)
    result.push_back(call->args[i]);
  return result;
}

ffi::Map<ffi::String, ffi::Any> ReadAnnotations(const ffi::Array<Expr> &args,
                                                const DictAttrs &attrs) {
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

ffi::Map<ffi::String, ffi::Any> ReadAnnotations(const CallObj *call) {
  if (call->op.same_as(layout_map_op())) {
    ReadLayoutMapAnnotation(call);
    return call->attrs.value().as_or_throw<DictAttrs>()->dict;
  }
  return ReadAnnotations(call->args,
                         call->attrs.value().as_or_throw<DictAttrs>());
}

ffi::Map<ffi::String, ffi::Any> ReadAnnotations(const RegionStmtObj *region) {
  return ReadAnnotations(region->args, region->attrs);
}

Stmt MakeAnnotation(const ffi::Map<ffi::String, ffi::Any> &annotations,
                    Location loc) {
  if (auto layouts = annotations.Get("layout_map")) {
    const auto marker = annotations.Get("tl.source_block_annotation");
    if (marker) {
      TVM_FFI_CHECK(marker->as<bool>().value_or(false), ValueError)
          << "tl.source_block_annotation must be the source annotation marker";
    }
    Stmt layout = MakeLayoutMapAnnotation(
        layouts->cast<ffi::Map<tirx::TensorVar, Layout>>(), marker.has_value(),
        loc);
    auto remaining = annotations;
    remaining.erase("layout_map");
    // Do not emit a marker-only second statement for a pure layout declaration.
    if (remaining.empty() || (remaining.size() == 1 &&
                              remaining.count("tl.source_block_annotation")))
      return layout;
    return SeqStmt({layout, MakeAnnotation(remaining, loc)}, loc);
  }
  auto encoded = EncodeAnnotations(annotations);
  return Evaluate(Call(PrimType::Void(), Op::Get("tl.annotation"),
                       {encoded.payload},
                       DictAttrs({{"schema_version", 1},
                                  {"annotation_schema", encoded.schema}}),
                       {}, loc),
                  loc);
}

RegionStmt MakeAttribute(ffi::Any node, ffi::String key, ffi::Any value,
                         SeqStmt body, Location loc) {
  auto encoded = EncodeAnnotations(ffi::Map<ffi::String, ffi::Any>{
      {"node", node}, {"key", key}, {"value", value}});
  return RegionStmt(
      Op::Get("tl.attr"), {encoded.payload}, {},
      DictAttrs({{"schema_version", 1}, {"annotation_schema", encoded.schema}}),
      std::move(body), {}, loc);
}

bool AnnotationAsBool(ffi::AnyView value) {
  if (auto boolean = value.as<bool>())
    return *boolean;
  if (const auto *literal = value.as<IntImmObj>()) {
    TVM_FFI_CHECK(literal->value == 0 || literal->value == 1, ValueError)
        << "Boolean annotation must be a boolean";
    return literal->value == 1;
  }
  auto integer = value.cast<int64_t>();
  TVM_FFI_CHECK(integer == 0 || integer == 1, ValueError)
      << "Boolean annotation must be a boolean";
  return integer != 0;
}

namespace {

using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;

// Source block annotations merge maps recursively, while leaves retain FFI
// value/identity equality. In particular, two separately constructed but
// structurally equal IR nodes or Arrays are not interchangeable declarations.
Map<Any, Any> MergeSourceAnnotations(const Map<Any, Any> &new_attrs,
                                     const Map<Any, Any> &old_attrs) {
  Map<Any, Any> result = old_attrs;
  for (const auto &[key, value] : new_attrs) {
    auto old = old_attrs.Get(key);
    if (!old.has_value()) {
      result.Set(key, value);
      continue;
    }
    auto old_map = old.value().as<Map<Any, Any>>();
    auto new_map = value.as<Map<Any, Any>>();
    if (old_map && new_map) {
      // Preserve the original nested-map traversal order as well as its
      // conflict rule; this is independent of the dynamic-operand IR codec.
      result.Set(key, MergeSourceAnnotations(old_map.value(), new_map.value()));
    } else if (!ffi::AnyEqual()(old.value(), value)) {
      TVM_FFI_THROW(ValueError)
          << "Try to merge two annotations with different values for key `"
          << key << "`, previous one is " << old.value() << ", new one is "
          << value;
    }
  }
  return result;
}

class AnnotationEncoder {
public:
  Array<Expr> values;

  int64_t Slot(Expr expression) {
    int64_t index = static_cast<int64_t>(values.size());
    values.push_back(std::move(expression));
    return index;
  }

  Array<Any> Encode(const Any &value) {
    if (auto expression = value.as<Expr>())
      return {"expr", Slot(*expression)};
    if (value == nullptr)
      return {"none"};
    if (auto boolean = value.as<bool>())
      return {"bool", *boolean};
    if (auto integer = value.as<ffi::BigInt>())
      return {"int", *integer};
    if (auto floating = value.as<double>())
      return {"float", *floating};
    if (auto string = value.as<String>())
      return {"str", *string};
    if (auto map = value.as<Map<Any, Any>>()) {
      Array<Any> schema{"map"};
      for (const auto &[key, element] : *map)
        schema.push_back(Array<Any>{Encode(key), Encode(element)});
      return schema;
    }
    if (auto array = value.as<Array<Any>>()) {
      Array<Any> schema{"list"};
      for (const auto &element : *array)
        schema.push_back(Encode(element));
      return schema;
    }
    if (auto layout = value.as<Layout>()) {
      String kind = LayoutPayloadKind(*layout);
      return {"layout", kind, 1, Slot(EncodeLayoutPayload(*layout))};
    }
    if (auto sync_kind = value.as<WSSyncKind>())
      return {"ws_sync_kind", (*sync_kind)->_str_index,
              sync_kind->CanonicalOrdinal()};
    if (auto role = value.as<WSRole>())
      return {"ws_role",
              Encode(Map<String, Any>{{"name", (*role)->name},
                                      {"warp_lo", (*role)->warp_lo},
                                      {"warp_hi", (*role)->warp_hi},
                                      {"max_nreg", (*role)->max_nreg}})};
    if (auto pipeline = value.as<WSPipeline>())
      return {"ws_pipeline",
              Encode(Map<String, Any>{{"name", (*pipeline)->name},
                                      {"buffers", (*pipeline)->buffers},
                                      {"depth", (*pipeline)->depth}})};
    if (auto op_ref = value.as<WSOpRef>())
      return {"ws_op_ref", Encode(Map<String, Any>{{"id", (*op_ref)->id}})};
    if (auto sync = value.as<WSSync>())
      return {"ws_sync",
              Encode(Map<String, Any>{{"kind", (*sync)->kind},
                                      {"pipeline", (*sync)->pipeline},
                                      {"stage", (*sync)->stage}})};
    if (auto scope = value.as<WSScope>())
      return {"ws_scope",
              Encode(Map<String, Any>{{"id", (*scope)->id},
                                      {"bodies", (*scope)->bodies}})};
    if (auto schedule = value.as<WSSchedule>())
      return {"ws_schedule",
              Encode(Map<String, Any>{{"num_warps", (*schedule)->num_warps},
                                      {"roles", (*schedule)->roles},
                                      {"pipelines", (*schedule)->pipelines},
                                      {"scopes", (*schedule)->scopes}})};
    TVM_FFI_THROW(TypeError)
        << "Unsupported TileLang annotation value: " << value.GetTypeKey();
  }
};

void SchemaArity(const Array<Any> &schema, size_t expected) {
  TVM_FFI_CHECK_EQ(schema.size(), expected, ValueError)
      << "Malformed TileLang annotation schema";
}

Expr PayloadAt(const Tuple &payload, const Any &index) {
  int64_t i = index.cast<int64_t>();
  TVM_FFI_CHECK(i >= 0 && static_cast<size_t>(i) < payload->fields.size(),
                ValueError)
      << "TileLang annotation operand index out of bounds: " << i;
  return payload->fields[i];
}

Any Decode(const Tuple &payload, const Array<Any> &schema) {
  TVM_FFI_CHECK(!schema.empty(), ValueError)
      << "Empty TileLang annotation schema";
  String kind = schema[0].cast<String>();
  if (kind == "none") {
    SchemaArity(schema, 1);
    return nullptr;
  }
  if (kind == "expr") {
    SchemaArity(schema, 2);
    return PayloadAt(payload, schema[1]);
  }
  if (kind == "bool" || kind == "int" || kind == "float" || kind == "str") {
    SchemaArity(schema, 2);
    if (kind == "bool")
      return schema[1].cast<bool>();
    if (kind == "int") {
      // BigInt accepts small and arbitrary-width integer metadata without a
      // lossy intermediate int64 conversion.
      (void)schema[1].cast<ffi::BigInt>();
      return schema[1];
    }
    if (kind == "float")
      return schema[1].cast<double>();
    return schema[1].cast<String>();
  }
  if (kind == "list" || kind == "tuple") {
    Array<Any> values;
    for (size_t i = 1; i < schema.size(); ++i) {
      values.push_back(Decode(payload, schema[i].cast<Array<Any>>()));
    }
    return values;
  }
  if (kind == "map") {
    Map<Any, Any> values;
    for (size_t i = 1; i < schema.size(); ++i) {
      auto pair = schema[i].cast<Array<Any>>();
      SchemaArity(pair, 2);
      Any key = Decode(payload, pair[0].cast<Array<Any>>());
      TVM_FFI_CHECK(!values.count(key), ValueError)
          << "Duplicate TileLang annotation key";
      values.Set(key, Decode(payload, pair[1].cast<Array<Any>>()));
    }
    return values;
  }
  if (kind == "layout") {
    SchemaArity(schema, 4);
    String layout_kind = schema[1].cast<String>();
    int64_t version = schema[2].cast<int64_t>();
    TVM_FFI_CHECK_EQ(version, 1, ValueError)
        << "Unsupported TileLang layout payload version";
    auto layout = PayloadAt(payload, schema[3]).as_or_throw<Tuple>();
    auto decode = ffi::Function::GetGlobalRequired("tl.DecodeLayoutPayload");
    return decode(layout, layout_kind, version);
  }
  if (kind == "ws_sync_kind") {
    SchemaArity(schema, 3);
    auto entry =
        ffi::EnumObj::_GetByStrIndex<WSSyncKindObj>(schema[1].cast<String>())
            .as_or_throw<WSSyncKind>();
    TVM_FFI_CHECK_EQ(entry.CanonicalOrdinal(), schema[2].cast<int64_t>(),
                     ValueError)
        << "Warp-specialization sync kind name and ordinal disagree";
    return entry;
  }
  if (kind == "ws_role" || kind == "ws_pipeline" || kind == "ws_op_ref" ||
      kind == "ws_sync" || kind == "ws_scope" || kind == "ws_schedule") {
    SchemaArity(schema, 2);
    auto decoded =
        Decode(payload, schema[1].cast<Array<Any>>()).cast<Map<Any, Any>>();
    Map<String, Any> fields;
    for (const auto &[key, value] : decoded)
      fields.Set(key.cast<String>(), value);
    if (kind == "ws_role") {
      TVM_FFI_CHECK_EQ(fields.size(), 4, ValueError);
      return WSRole(fields.at("name").cast<String>(),
                    fields.at("warp_lo").cast<int64_t>(),
                    fields.at("warp_hi").cast<int64_t>(),
                    fields.at("max_nreg").cast<int64_t>());
    }
    if (kind == "ws_pipeline") {
      TVM_FFI_CHECK_EQ(fields.size(), 3, ValueError);
      return WSPipeline(fields.at("name").cast<String>(),
                        fields.at("buffers").cast<Array<tirx::TensorVar>>(),
                        fields.at("depth").cast<int64_t>());
    }
    if (kind == "ws_op_ref") {
      TVM_FFI_CHECK_EQ(fields.size(), 1, ValueError);
      return WSOpRef(fields.at("id").cast<String>());
    }
    if (kind == "ws_sync") {
      TVM_FFI_CHECK_EQ(fields.size(), 3, ValueError);
      return WSSync(fields.at("kind").cast<WSSyncKind>(),
                    fields.at("pipeline").cast<String>(),
                    fields.at("stage").cast<int64_t>());
    }
    if (kind == "ws_scope") {
      TVM_FFI_CHECK_EQ(fields.size(), 2, ValueError);
      return WSScope(fields.at("id").cast<String>(),
                     fields.at("bodies").cast<Map<String, Array<WSInstr>>>());
    }
    TVM_FFI_CHECK_EQ(fields.size(), 4, ValueError);
    return WSSchedule(fields.at("num_warps").cast<int64_t>(),
                      fields.at("roles").cast<Array<WSRole>>(),
                      fields.at("pipelines").cast<Array<WSPipeline>>(),
                      fields.at("scopes").cast<Array<WSScope>>());
  }
  TVM_FFI_THROW(ValueError)
      << "Unknown TileLang annotation schema tag: " << kind;
}

void ValidateSchema(const DictAttrs &attrs) {
  TVM_FFI_CHECK_EQ(attrs.GetAttr<int64_t>("schema_version", 0).value(), 1,
                   ValueError)
      << "Unsupported TileLang annotation version";
  TVM_FFI_CHECK(attrs.GetAttr<Array<Any>>("annotation_schema").has_value(),
                ValueError)
      << "Missing annotation_schema";
}

DictAttrs GetSchemaAttrs(const CallObj *call) {
  TVM_FFI_CHECK(call->attrs.has_value(), ValueError)
      << "TileLang annotation schema is required";
  DictAttrs attrs = call->attrs.value().as_or_throw<DictAttrs>();
  ValidateSchema(attrs);
  return attrs;
}

void ValidateAnnotationCall(const CallObj *call) {
  auto attrs = GetSchemaAttrs(call);
  TVM_FFI_CHECK_EQ(call->args.size(), 1, ValueError);
  TVM_FFI_CHECK(call->ty_args.empty(), ValueError)
      << "Annotation calls have no type arguments";
  Decode(call->args[0].as_or_throw<Tuple>(),
         attrs.GetAttr<Array<Any>>("annotation_schema").value());
}

Array<Var> AttributeBodyParams(const CallObj *call) {
  ValidateAnnotationCall(call);
  return {};
}

void ValidateAttributeRegion(const RegionStmtObj *region) {
  ValidateSchema(region->attrs);
  TVM_FFI_CHECK_EQ(region->args.size(), 1, ValueError);
  Decode(region->args[0].as_or_throw<Tuple>(),
         region->attrs.GetAttr<Array<Any>>("annotation_schema").value());
  TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                ValueError)
      << "tl.attr has no body parameters or results";
}

Map<Any, Any> MergeDecodedSourceAnnotations(Map<Any, Any> result,
                                            const Map<Any, Any> &values) {
  for (const auto &[key, value] : values) {
    auto previous = result.Get(key);
    auto old_map = previous ? previous->as<Map<Any, Any>>() : std::nullopt;
    auto new_map = value.as<Map<Any, Any>>();
    if (old_map && new_map) {
      result.Set(key, MergeDecodedSourceAnnotations(*old_map, *new_map));
    } else {
      // Construction already checked source FFI identity conflicts. Decoding
      // a layout creates a fresh object, so rechecking AnyEqual here would
      // reject two reads of the same valid declaration.
      result.Set(key, value);
    }
  }
  return result;
}

void CollectSourceAnnotations(const Stmt &statement, Map<Any, Any> *values) {
  if (const auto *sequence = statement.as<SeqStmtObj>()) {
    for (const Stmt &child : sequence->seq)
      CollectSourceAnnotations(child, values);
  } else if (const auto *region = statement.as<RegionStmtObj>()) {
    if (region->op->name == "tl.launch_kernel" ||
        ascend::IsAscendBlock(region) || IsLexicalScope(region))
      return;
    if (region->op->name == "tl.attr") {
      ValidateSchema(region->attrs);
      const auto fields =
          Decode(region->args[0].as_or_throw<Tuple>(),
                 region->attrs.GetAttr<Array<Any>>("annotation_schema").value())
              .cast<Map<Any, Any>>();
      const auto key = fields.at(String("key")).cast<String>();
      if (key == "tl.kernel")
        return;
    }
    CollectSourceAnnotations(region->body, values);
  } else if (const auto *loop = statement.as<ForObj>()) {
    CollectSourceAnnotations(loop->body, values);
  } else if (const auto *loop = statement.as<WhileObj>()) {
    CollectSourceAnnotations(loop->body, values);
  } else if (const auto *branch = statement.as<IfObj>()) {
    CollectSourceAnnotations(branch->then_case, values);
    if (branch->else_case)
      CollectSourceAnnotations(*branch->else_case, values);
  } else if (const auto *evaluate = statement.as<EvaluateObj>()) {
    const auto *call = evaluate->value.as<CallObj>();
    if (!IsAnnotationCall(call))
      return;
    Map<Any, Any> annotations;
    for (const auto &[key, value] : ReadAnnotations(call))
      annotations.Set(key, value);
    auto source = annotations.Get(String("tl.source_block_annotation"));
    if (!source)
      return;
    TVM_FFI_CHECK(source->as<bool>().value_or(false), ValueError)
        << "tl.source_block_annotation must be the source annotation marker";
    annotations.erase(String("tl.source_block_annotation"));
    *values = MergeDecodedSourceAnnotations(*values, annotations);
  }
  // Do not walk expressions: a Bind may hold an independent Function/Lambda,
  // and metadata operands are uses, never child statement scopes.
}

} // namespace

ffi::Any DecodeAnnotations(const Tuple &payload,
                           const ffi::Array<ffi::Any> &schema) {
  return Decode(payload, schema);
}

EncodedAnnotations EncodeAnnotations(const ffi::Any &annotations) {
  AnnotationEncoder encoder;
  auto schema = encoder.Encode(annotations);
  return {Tuple(std::move(encoder.values)), std::move(schema)};
}

bool IsStaticAnnotationValue(const Any &value) {
  if (value == nullptr || value.as<bool>() || value.as<ffi::BigInt>() ||
      value.as<double>() || value.as<String>() || value.as<IntImmObj>() ||
      value.as<FloatImmObj>() || value.as<StringImmObj>())
    return true;
  if (auto array = value.as<Array<Any>>()) {
    for (const Any &item : *array)
      if (!IsStaticAnnotationValue(item))
        return false;
    return true;
  }
  if (auto map = value.as<Map<Any, Any>>()) {
    for (const auto &[key, item] : *map)
      if (!IsStaticAnnotationValue(key) || !IsStaticAnnotationValue(item))
        return false;
    return true;
  }
  return false;
}

bool IsCodegenAnnotation(const String &key) {
  // Only keys with concrete codegen or outlined-source consumers belong here.
  // Target-specific validation remains with the corresponding backend.
  return key == "pragma_import_c" || key == "pragma_unroll_factor" ||
         key == "code_block_source" || key == "code_block_signature" ||
         key == "code_block_entry_name" || key == "volatile_scope" ||
         key == "fragment_shape" || key == "fragment_layout" ||
         key == "threadblock_swizzle_pattern" || key == "tl.min_blocks_per_sm";
}

bool IsCompilerAnnotation(const String &key) {
  // Reserved namespaces are compiler contracts even when a new owner has not
  // yet been added. The remaining names are legacy source grammar and known
  // execution options, not a list of owning passes or pipeline stages.
  if (key.find("tl.") == 0 || key.find("tirx.") == 0)
    return true;
  static const std::unordered_set<std::string> contracts{
      "barrier_init",
      "l2_hit_ratio_map",
      "metal_context",
      "cluster_dims",
      "use_2cta",
      "warp_specialize",
      "kWarpSpecializationScope",
      "kCustomWarpSpecialization",
      "thread_extent",
      "virtual_thread",
      "coproc_scope",
      "pipeline_exec_scope",
      "reduce_scope",
      "vector_count",
      "tilelang_assume",
      "pragma_auto_unroll_max_step",
      "pragma_unroll_explicit",
      "pragma_debug_skip_region",
      "layout_map",
      "safe_value_map",
      "precision",
      "round",
      "sat"};
  return contracts.count(key);
}

Call NormalizeCallAnnotations(Call call) {
  if (const auto *intrinsic = call->op.as<OpObj>()) {
    TVM_FFI_CHECK(intrinsic->name.find("tl.tileop.") != 0, ValueError)
        << "LowerSourceAnnotations cannot run before LowerTileOp: "
        << intrinsic->name;
  }
  if (!call->attrs)
    return call;
  auto attrs = call->attrs.value().as<DictAttrs>();
  if (!attrs || !attrs.value()->dict.count("operand_count"))
    return call;

  // Keep runtime operands separate from source metadata. The shared intrinsic
  // decoder validates the complete envelope before any payload is discarded.
  auto operands = DecodeCallOperands(call.get());
  auto decoded =
      DecodeAnnotations(
          call->args.back().as_or_throw<Tuple>(),
          attrs.value().GetAttr<Array<Any>>("annotation_schema").value())
          .cast<Map<String, Any>>();
  auto attributes = attrs.value()->dict;
  attributes.erase("schema_version");
  attributes.erase("operand_count");
  attributes.erase("annotation_schema");
  std::vector<String> source_keys;
  for (const auto &[key, value] : decoded) {
    TVM_FFI_CHECK(!attributes.count(key), ValueError)
        << "Annotation appears in both native attributes and metadata "
           "operands: "
        << key;
    if (IsStaticAnnotationValue(value)) {
      attributes.Set(key, value);
      source_keys.push_back(key);
    } else {
      TVM_FFI_CHECK(!IsCodegenAnnotation(key) && !IsCompilerAnnotation(key),
                    ValueError)
          << "Unconsumed intrinsic annotation `" << key
          << "` at LowerSourceAnnotations; lower its value to an explicit "
             "operand or a compile-time option first";
      // Unknown descriptive hints are not runtime arguments. Removing their
      // payload must not introduce an evaluation of the discarded expression.
    }
  }
  if (!source_keys.empty()) {
    std::sort(source_keys.begin(), source_keys.end());
    attributes.Set("tl.source_annotation_keys", Array<String>(source_keys));
  }
  return Call(call->ty, call->op, std::move(operands),
              attributes.empty() ? ffi::Optional<Attrs>()
                                 : ffi::Optional<Attrs>(DictAttrs(attributes)),
              call->ty_args, call->loc);
}

Map<String, Any> CollectSourceBlockAnnotations(const SeqStmt &body) {
  Map<Any, Any> collected;
  CollectSourceAnnotations(body, &collected);
  Map<String, Any> result;
  for (const auto &[key, value] : collected)
    result.Set(key.cast<String>(), value);
  return result;
}

Call MakeAnnotatedCall(Type ty, Op op, Array<Expr> operands,
                       const Map<String, Any> &annotations, Location loc) {
  if (op->name.find("tl.tileop.") != 0 &&
      IsStaticAnnotationValue(annotations)) {
    Map<String, Any> attributes = annotations;
    if (!annotations.empty()) {
      std::vector<String> keys;
      for (const auto &[key, value] : annotations) {
        TVM_FFI_CHECK(key != "schema_version" && key != "operand_count" &&
                          key != "annotation_schema" &&
                          key != "tl.source_annotation_keys",
                      ValueError)
            << "Reserved intrinsic annotation key: " << key;
        keys.push_back(key);
      }
      // Shape comparison ignores source annotations, but must retain attributes
      // supplied directly through the native IR API. Keep that provenance
      // without adding an expression operand or a metadata payload.
      std::sort(keys.begin(), keys.end());
      attributes.Set("tl.source_annotation_keys", Array<String>(keys));
    }
    return Call(std::move(ty), std::move(op), std::move(operands),
                annotations.empty()
                    ? ffi::Optional<Attrs>()
                    : ffi::Optional<Attrs>(DictAttrs(attributes)),
                {}, std::move(loc));
  }
  auto encoded = EncodeAnnotations(annotations);
  const int64_t count = static_cast<int64_t>(operands.size());
  operands.push_back(encoded.payload);
  DictAttrs attrs(Map<String, Any>{{"schema_version", 1},
                                   {"operand_count", count},
                                   {"annotation_schema", encoded.schema}});
  return Call(std::move(ty), std::move(op), std::move(operands),
              std::move(attrs), {}, std::move(loc));
}

TileOpArguments DecodeTileOpArguments(const CallObj *call) {
  DictAttrs attrs = GetSchemaAttrs(call);
  int64_t count = attrs.GetAttr<int64_t>("operand_count", -1).value();
  TVM_FFI_CHECK(count >= 0 &&
                    static_cast<size_t>(count + 1) == call->args.size(),
                ValueError)
      << "A TileLang tile operation needs its declared operands and one "
         "annotation Tuple";
  TileOpArguments result;
  for (int64_t i = 0; i < count; ++i)
    result.operands.push_back(call->args[i]);
  Any decoded = Decode(call->args.back().as_or_throw<Tuple>(),
                       attrs.GetAttr<Array<Any>>("annotation_schema").value());
  auto annotations = decoded.cast<Map<Any, Any>>();
  for (const auto &[key, value] : annotations) {
    result.annotations.Set(key.cast<String>(), value);
  }
  return result;
}

namespace {
void ValidateLayoutMapCall(const CallObj *call) {
  ReadLayoutMapAnnotation(call);
}

ffi::Optional<script::printer::ExprDoc>
LayoutMapCallDoc(script::printer::DocTranslatorObj *translator,
                 ffi::AnyView value, const ffi::Object *) {
  // This dialect has no global `tl` script namespace. The native constructor
  // preserves the typed attrs and works with the standard TIRx parser.
  return script::printer::RawCallDocTranslate(translator,
                                              value.cast<Call>().get());
}
} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.MergeSourceAnnotations", MergeSourceAnnotations)
      .def("tl.CollectSourceBlockAnnotations", CollectSourceBlockAnnotations)
      .def("tl.MakeAnnotation", MakeAnnotation)
      .def("tl.ReadAnnotations",
           [](const Call &call) { return ReadAnnotations(call.get()); })
      .def("tl.MakeLayoutMapAnnotation", MakeLayoutMapAnnotation)
      .def("tl.ReadLayoutMapAnnotation",
           [](const Call &call) { return ReadLayoutMapAnnotation(call.get()); })
      .def("tl.DecodeAnnotations", DecodeAnnotations)
      .def("tl.EncodeAnnotations",
           [](Any value) {
             auto encoded = EncodeAnnotations(value);
             return Array<Any>{encoded.schema, encoded.payload};
           })
      .def("tl.MakeAnnotatedCall", MakeAnnotatedCall);
  OpDef("tl.layout_map", "Typed TileLang layout declarations.")
      .set_attr(script::printer::op_attr::kDocTranslateOpCall,
                script::printer::FDocTranslate::FromNative<&LayoutMapCallDoc>())
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateLayoutMapCall>())
      .signature(sig::call_attrs<DictAttrsObj>())
      .set_attr(op_attr::kFixedReturnType, PrimType::Void())
      .set_attr(op_attr::kCallEffectKind, CallEffectKind::kOpaque);
  OpDef("tl.annotation", "Metadata for the enclosing TileLang lexical scope.")
      .set_validator(
          ffi::reflection::NativeFunctionView<void(
              const CallObj *)>::FromNative<&ValidateAnnotationCall>())
      .signature(sig::arg<Tuple>("payload"), sig::call_attrs<DictAttrsObj>())
      .set_attr(op_attr::kFixedReturnType, PrimType::Void())
      // Generic simplification must not erase metadata before its TL consumer.
      .set_attr(op_attr::kCallEffectKind, CallEffectKind::kOpaque);
  OpDef("tl.attr",
        "A TileLang attribute scope with explicit metadata operands.")
      .signature(sig::arg<Tuple>("payload"), sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&AttributeBodyParams>(op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateAttributeRegion>(op_attr::kRegionValidate);
}

} // namespace tvm::tl
