/*!
 * \file tl/ir/annotations.cc
 * \brief Explicit dynamic operands for TileLang compile-time metadata.
 */

#include "annotations.h"
#include "layout/payload.h"
#include "transform/common/warp_specialize.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>
#include <tvm/ir/stmt.h>

namespace tvm::tl {
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
    if (region->op->name == "tl.launch_kernel")
      return;
    if (region->op->name == "tl.attr") {
      ValidateSchema(region->attrs);
      const auto fields =
          Decode(region->args[0].as_or_throw<Tuple>(),
                 region->attrs.GetAttr<Array<Any>>("annotation_schema").value())
              .cast<Map<Any, Any>>();
      const auto key = fields.at(String("key")).cast<String>();
      if (key == "tl.kernel" || key == "tl.sblock")
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
    if (!call || !call->op.same_as(Op::Get("tl.annotation")))
      return;
    auto attrs = GetSchemaAttrs(call);
    auto annotations =
        Decode(call->args[0].as_or_throw<Tuple>(),
               attrs.GetAttr<Array<Any>>("annotation_schema").value())
            .cast<Map<Any, Any>>();
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

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef()
      .def("tl.MergeSourceAnnotations", MergeSourceAnnotations)
      .def("tl.CollectSourceBlockAnnotations", CollectSourceBlockAnnotations)
      .def("tl.DecodeAnnotations", DecodeAnnotations)
      .def("tl.EncodeAnnotations",
           [](Any value) {
             auto encoded = EncodeAnnotations(value);
             return Array<Any>{encoded.schema, encoded.payload};
           })
      .def("tl.MakeAnnotatedCall", MakeAnnotatedCall);
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
