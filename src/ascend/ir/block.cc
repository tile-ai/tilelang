/*!
 * \file ascend/ir/block.cc
 * \brief Native CUBE, VECTOR and VF execution-region metadata.
 */

#include "block.h"
#include "ir/annotations.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/op.h>

namespace tvm::tl::ascend {
namespace {

using ffi::Any;
using ffi::Array;
using ffi::Map;
using ffi::String;
using Annotations = Map<String, Any>;

bool IsStaticMetadata(const Any &value) {
  if (value == nullptr || value.as<bool>() || value.as<ffi::BigInt>() ||
      value.as<double>() || value.as<String>() || value.as<IntImmObj>() ||
      value.as<FloatImmObj>() || value.as<StringImmObj>())
    return true;
  if (auto array = value.as<Array<Any>>()) {
    for (const Any &element : *array)
      if (!IsStaticMetadata(element))
        return false;
    return true;
  }
  if (auto map = value.as<Map<Any, Any>>()) {
    for (const auto &[key, element] : *map)
      if (!IsStaticMetadata(key) || !IsStaticMetadata(element))
        return false;
    return true;
  }
  return false;
}

Annotations ReadBlockAnnotations(const Array<Expr> &args,
                                 const DictAttrs &attrs) {
  auto name = attrs.GetAttr<String>("name");
  TVM_FFI_CHECK(name && IsAscendBlockName(*name), ValueError)
      << "tl.ascend_block name must be CUBE, VECTOR, SIMT_VF or SIMD_VF";
  for (const auto &[key, value] : attrs->dict) {
    TVM_FFI_CHECK(key == "name" || key == "annotations" ||
                      key == "annotation_schema",
                  ValueError)
        << "Unknown tl.ascend_block attribute: " << key;
    TVM_FFI_CHECK(IsStaticMetadata(value), TypeError)
        << "tl.ascend_block attrs must contain only static metadata; "
           "put expression-dependent annotations in explicit operands";
  }
  auto constants = attrs.GetAttr<Annotations>("annotations");
  TVM_FFI_CHECK(constants.has_value(), ValueError)
      << "tl.ascend_block requires an annotations map";
  Annotations annotations = *constants;
  auto schema = attrs.GetAttr<Array<Any>>("annotation_schema");
  TVM_FFI_CHECK_EQ(args.size(), schema ? 1U : 0U, ValueError)
      << "tl.ascend_block accepts one metadata Tuple exactly when an "
         "annotation_schema is present";
  if (schema) {
    auto dynamic = DecodeAnnotations(args[0].as_or_throw<Tuple>(), *schema)
                       .cast<Annotations>();
    for (const auto &[key, value] : dynamic) {
      TVM_FFI_CHECK(!annotations.count(key), ValueError)
          << "Duplicate static and dynamic Ascend block annotation: " << key;
      annotations.Set(key, value);
    }
  }
  return annotations;
}

struct EncodedBlock {
  Array<Expr> args;
  DictAttrs attrs;
};

EncodedBlock EncodeBlock(String name, const Annotations &annotations) {
  TVM_FFI_CHECK(IsAscendBlockName(name), ValueError)
      << "tl.ascend_block name must be CUBE, VECTOR, SIMT_VF or SIMD_VF";
  Annotations constants, dynamic;
  for (const auto &[key, value] : annotations) {
    if (IsStaticMetadata(value))
      constants.Set(key, value);
    else
      dynamic.Set(key, value);
  }
  Annotations attrs{{"name", std::move(name)}, {"annotations", constants}};
  Array<Expr> args;
  if (!dynamic.empty()) {
    auto encoded = EncodeAnnotations(dynamic);
    args.push_back(encoded.payload);
    attrs.Set("annotation_schema", encoded.schema);
  }
  return {std::move(args), DictAttrs(std::move(attrs))};
}

Array<Var> BlockBodyParams(const CallObj *call) {
  TVM_FFI_CHECK(call->ty_args.empty(), ValueError)
      << "tl.ascend_block has no type arguments";
  ReadBlockAnnotations(call->args,
                       call->attrs.value().as_or_throw<DictAttrs>());
  return {};
}

void ValidateBlock(const RegionStmtObj *region) {
  TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                ValueError)
      << "tl.ascend_block has no body parameters or results";
  ReadBlockAnnotations(region->args, region->attrs);
}

} // namespace

bool IsAscendBlockName(const String &name) {
  return name == "CUBE" || name == "VECTOR" || name == "SIMT_VF" ||
         name == "SIMD_VF";
}

bool IsAscendBlock(const RegionStmtObj *region) {
  return region && region->op->name == "tl.ascend_block";
}

String AscendBlockName(const RegionStmtObj *region) {
  TVM_FFI_CHECK(IsAscendBlock(region), ValueError)
      << "Expected a tl.ascend_block region";
  return region->attrs.GetAttr<String>("name").value();
}

Annotations AscendBlockAnnotations(const RegionStmtObj *region) {
  TVM_FFI_CHECK(IsAscendBlock(region), ValueError)
      << "Expected a tl.ascend_block region";
  return ReadBlockAnnotations(region->args, region->attrs);
}

// Lexical VF markers and source-block annotations are still emitted by the
// frontend. Keep their interpretation at the Ascend execution-scope boundary.
VFKind GetVFKind(const RegionStmtObj *region) {
  if (IsAscendBlock(region)) {
    const auto name = AscendBlockName(region);
    bool simd = name == "SIMD_VF";
    bool simt = name == "SIMT_VF";
    const auto annotations = AscendBlockAnnotations(region);
    if (auto value = annotations.Get("tl.simdvf_scope"))
      simd |= AnnotationAsBool(*value);
    if (auto value = annotations.Get("tl.simtvf_scope"))
      simt |= AnnotationAsBool(*value);
    TVM_FFI_CHECK(!(simd && simt), ValueError)
        << "An Ascend scope cannot be both SIMD_VF and SIMT_VF";
    return simd ? VFKind::kSimd : simt ? VFKind::kSimt : VFKind::kNone;
  }
  if (!region || region->op->name != "tl.attr")
    return VFKind::kNone;
  TVM_FFI_CHECK_EQ(region->args.size(), 1U, ValueError);
  TVM_FFI_CHECK_EQ(region->attrs.GetAttr<int64_t>("schema_version").value_or(0),
                   1, ValueError);
  auto fields = ReadAnnotations(region);
  auto string_value = [](ffi::AnyView value) {
    if (const auto *literal = value.as<StringImmObj>())
      return literal->value;
    return value.cast<String>();
  };
  const auto key = string_value(fields.at("key"));
  bool simd = key == "tl.simdvf_scope";
  bool simt = key == "tl.simtvf_scope";
  if (key == "tl.sblock") {
    auto block = fields.at("value").cast<ffi::Map<ffi::String, ffi::Any>>();
    const auto name = string_value(block.at("name"));
    simd = name == "SIMD_VF";
    simt = name == "SIMT_VF";
    if (auto metadata = block.Get("annotations")) {
      auto annotations = metadata->cast<ffi::Map<ffi::String, ffi::Any>>();
      if (auto value = annotations.Get("tl.simdvf_scope"))
        simd |= AnnotationAsBool(*value);
      if (auto value = annotations.Get("tl.simtvf_scope"))
        simt |= AnnotationAsBool(*value);
    }
  }
  TVM_FFI_CHECK(!(simd && simt), ValueError)
      << "An Ascend scope cannot be both SIMD_VF and SIMT_VF";
  if (simd || simt) {
    TVM_FFI_CHECK(region->body_params.empty() && region->result_vars.empty(),
                  ValueError)
        << "Ascend VF metadata cannot introduce hidden region bindings";
  }
  return simd ? VFKind::kSimd : simt ? VFKind::kSimt : VFKind::kNone;
}

RegionStmt MakeAscendBlock(String name, const Annotations &annotations,
                           SeqStmt body, Location loc) {
  auto encoded = EncodeBlock(std::move(name), annotations);
  return RegionStmt(Op::Get("tl.ascend_block"), std::move(encoded.args), {},
                    std::move(encoded.attrs), std::move(body), {},
                    std::move(loc));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  OpDef("tl.ascend_block",
        "An Ascend core or vector-function execution region.")
      .signature(sig::var_args<Tuple>("annotations"),
                 sig::call_attrs<DictAttrsObj>())
      .set_native_fn_attr<&BlockBodyParams>(op_attr::kRegionGetBodyParams)
      .set_native_fn_attr<&ValidateBlock>(op_attr::kRegionValidate);
  ffi::reflection::GlobalDef().def(
      "tl.EncodeAscendBlock", [](String name, const Annotations &annotations) {
        auto encoded = EncodeBlock(std::move(name), annotations);
        return Array<Any>{encoded.args, encoded.attrs};
      });
}

} // namespace tvm::tl::ascend
